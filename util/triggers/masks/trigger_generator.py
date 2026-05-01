"""
make_trigger_tier1.py
─────────────────────────────────────────────────────────────────────────────
Tier 1 — Target-Empirical Sampling ("The Mimic")

Colours every black pixel in a binary 56×56 trigger mask with an RGB value
sampled uniformly at random from the actual "Bus" pixels found in the
Cityscapes gtFine + leftImg8bit training split.

Usage
─────
    python make_trigger_tier1.py \
        --trigger      path/to/trigger_mask.png         \
        --cityscapes   /path/to/cityscapes/root         \
        --output       trigger_tier1.png                \
        [--split       train]                           \
        [--seed        42]                              \
        [--max-images  0]                               \    # 0 = all images
        [--alpha]                                            # keep background transparent

Cityscapes directory layout expected
─────────────────────────────────────
  <cityscapes_root>/
      gtFine/
          train/
              <city>/
                  <city>_<seq>_<frame>_gtFine_labelIds.png
      leftImg8bit/
          train/
              <city>/
                  <city>_<seq>_<frame>_leftImg8bit.png

Bus label
─────────
  Cityscapes labelId for "bus" = 28  (full labelIds 0-33 range)
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Cityscapes constants ──────────────────────────────────────────────────────
ROAD_LABEL_ID = 7           # Change this to the target label ID you want
TRIGGER_BLACK_THRESHOLD = 10  # pixel value ≤ this is treated as "black / active"


# ── helpers ───────────────────────────────────────────────────────────────────

def collect_bus_pixels(cityscapes_root: Path, split: str, max_images: int, save_example: Path = None) -> np.ndarray:
    """
    Walk gtFine/<split> and collect every RGB pixel in leftImg8bit that
    corresponds to a Bus (labelId == 28) pixel in the annotation mask.

    Returns
    -------
    np.ndarray of shape (N, 3), dtype uint8
        All sampled bus RGB values concatenated across every image.
    """
    gt_root  = cityscapes_root / "gtFine"      / split
    img_root = cityscapes_root / "leftImg8bit" / split

    label_files = sorted(gt_root.rglob("*_gtFine_labelIds.png"))
    if not label_files:
        sys.exit(
            f"[ERROR] No gtFine_labelIds.png files found under {gt_root}.\n"
            "        Check --cityscapes path and --split value."
        )

    if max_images > 0:
        label_files = label_files[:max_images]

    all_bus_pixels: list[np.ndarray] = []
    example_saved = False
    first_image_labels = None

    print(f"[INFO] Scanning {len(label_files)} annotation files for Bus pixels …")
    for label_path in tqdm(label_files, unit="img"):
        # Derive matching leftImg8bit path from label path
        # e.g.  gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
        #     → leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
        stem = label_path.stem.replace("_gtFine_labelIds", "")
        city = label_path.parent.name
        img_path = img_root / city / f"{stem}_leftImg8bit.png"

        if not img_path.exists():
            tqdm.write(f"[WARN] Missing image: {img_path} — skipping")
            continue

        label_arr = np.array(Image.open(label_path))      # H×W, uint8 or int32
        rgb_arr   = np.array(Image.open(img_path).convert("RGB"))  # H×W×3

        # On first image, save all unique labels for reference
        if first_image_labels is None:
            first_image_labels = np.unique(label_arr)
            tqdm.write(f"[DEBUG] Unique label IDs in first image: {first_image_labels.tolist()}")
            tqdm.write(f"[DEBUG] Label range: {first_image_labels.min()} to {first_image_labels.max()}")
            if first_image_labels.max() <= 18:
                tqdm.write(f"[DEBUG] Detected trainIds (0-18 range). Bus = 15")
            elif first_image_labels.max() > 18:
                tqdm.write(f"[DEBUG] Detected full labelIds (0-33 range). Bus = 28")

        bus_mask = label_arr == ROAD_LABEL_ID
        if not bus_mask.any():
            continue

        # Save example image on first bus encounter
        if save_example and not example_saved:
            example_img = Image.open(img_path)
            example_img.save(save_example)
            
            # Also create a visualization showing where the bus pixels are
            vis_arr = rgb_arr.copy()
            vis_arr[bus_mask] = [0, 255, 0]  # Highlight bus pixels in green
            vis_img = Image.fromarray(vis_arr.astype(np.uint8), mode="RGB")
            vis_path = save_example.parent / f"bus_example_MASK_{save_example.name}"
            vis_img.save(vis_path)
            tqdm.write(f"[INFO] Saved example image with bus to {save_example}")
            tqdm.write(f"[INFO] Saved mask visualization to {vis_path} (bus pixels in green)")
            tqdm.write(f"[INFO] Bus pixels in this image: {bus_mask.sum()}")
            example_saved = True

        bus_pixels = rgb_arr[bus_mask]                     # shape (k, 3)
        all_bus_pixels.append(bus_pixels)

    if not all_bus_pixels:
        sys.exit(
            "[ERROR] No Bus pixels found in the dataset.\n"
            "        Are there any images with buses in the chosen split?"
        )

    pool = np.concatenate(all_bus_pixels, axis=0)          # (N, 3)
    print(f"[INFO] Collected {len(pool):,} Bus pixels from {len(all_bus_pixels)} images.")
    return pool


def load_binary_trigger(path: Path) -> np.ndarray:
    """
    Load the trigger mask and return a boolean array (H, W) where True = active
    pixel (originally black, i.e. value ≤ TRIGGER_BLACK_THRESHOLD).
    """
    img = Image.open(path)
    print(f"[DEBUG] Original image mode: {img.mode}, size: {img.size}")
    
    # If image has alpha, keep it separate
    if img.mode in ("RGBA", "LA", "PA"):
        img_rgb = img.convert("RGB")
        img_alpha = img.split()[-1]
        print(f"[DEBUG] Image has alpha channel")
        arr = np.array(img_rgb.convert("L"))
        # Treat transparent pixels (alpha < 128) as non-active
        alpha_arr = np.array(img_alpha)
        mask = (arr <= TRIGGER_BLACK_THRESHOLD) & (alpha_arr >= 128)
    else:
        img = img.convert("L")
        arr = np.array(img)
        mask = arr <= TRIGGER_BLACK_THRESHOLD
    
    n_active = mask.sum()
    if n_active == 0:
        sys.exit("[ERROR] No active (black) pixels found in the trigger mask.")
    print(f"[INFO] Trigger mask: {arr.shape[1]}×{arr.shape[0]}, {n_active} active pixels.")
    print(f"[DEBUG] Pixel values in image: min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")
    return mask


def colour_trigger(mask: np.ndarray, pool: np.ndarray, rng: np.random.Generator) -> Image.Image:
    """
    For every active pixel in *mask*, sample one RGB triple from *pool*
    uniformly at random (with replacement).

    Returns an RGBA PIL image:  active pixels = sampled colour (A=255),
                                 inactive pixels = transparent (A=0).
    """
    h, w = mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)              # RGBA, default transparent

    n_active = int(mask.sum())
    # Sample indices into pool with replacement
    indices = rng.integers(0, len(pool), size=n_active)
    sampled_colours = pool[indices]                        # (n_active, 3)

    out[mask, :3] = sampled_colours
    out[mask,  3] = 255                                    # fully opaque

    return Image.fromarray(out, mode="RGBA")


def save_trigger(img: Image.Image, output_path: Path, alpha: bool):
    """Save as RGBA (transparent BG) or RGB (white BG) PNG."""
    if alpha:
        img.save(output_path)
        print(f"[INFO] Saved RGBA trigger → {output_path}")
    else:
        # Composite over white background
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        bg.save(output_path)
        print(f"[INFO] Saved RGB trigger (white BG) → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Tier-1 empirical trigger: colour a binary mask using Bus "
                    "pixels sampled from Cityscapes."
    )
    p.add_argument("--trigger",    required=True, type=Path,
                   help="Path to the 56×56 binary trigger mask (PNG).")
    p.add_argument("--cityscapes", required=True, type=Path,
                   help="Root of the Cityscapes dataset (contains gtFine/ and leftImg8bit/).")
    p.add_argument("--output",     default="trigger_tier1.png", type=Path,
                   help="Output path for the coloured trigger (default: trigger_tier1.png).")
    p.add_argument("--split",      default="train",
                   choices=["train", "val", "test"],
                   help="Cityscapes split to mine for Bus pixels (default: train).")
    p.add_argument("--seed",       default=42, type=int,
                   help="Random seed for reproducibility (default: 42).")
    p.add_argument("--max-images", default=0, type=int,
                   help="Cap on annotation files to scan (0 = unlimited, default: 0).")
    p.add_argument("--alpha",      action="store_true",
                   help="Save with transparent background (RGBA). Default: white background.")
    p.add_argument("--save-example", type=Path, default=None,
                   help="Save an example image containing bus pixels (for verification).")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Seed both numpy and Python RNG
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    # 1. Load binary mask
    mask = load_binary_trigger(args.trigger)

    # 2. Harvest Bus pixels from Cityscapes
    pool = collect_bus_pixels(args.cityscapes, args.split, args.max_images, args.save_example)

    # 3. Colour the trigger
    coloured = colour_trigger(mask, pool, rng)

    # 4. Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_trigger(coloured, args.output, args.alpha)

    # 5. Quick sanity stats
    rgb = np.array(coloured)[mask]
    print(
        f"\n[STATS] Active-pixel colour summary (R | G | B)\n"
        f"        mean  : {rgb[:,0].mean():.1f} | {rgb[:,1].mean():.1f} | {rgb[:,2].mean():.1f}\n"
        f"        std   : {rgb[:,0].std():.1f}  | {rgb[:,1].std():.1f}  | {rgb[:,2].std():.1f}\n"
        f"        min   : {rgb[:,0].min()}  | {rgb[:,1].min()}  | {rgb[:,2].min()}\n"
        f"        max   : {rgb[:,0].max()}  | {rgb[:,1].max()}  | {rgb[:,2].max()}"
    )


if __name__ == "__main__":
    main()