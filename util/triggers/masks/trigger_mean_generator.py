"""
trigger_mean_generator.py
─────────────────────────────────────────────────────────────────────────────
Mean-Colored Trigger Generator

Creates triggers by coloring black pixels in multiple trigger masks with the
MEAN color of a target class. All pixels use the same mean color (no randomness).

Processes 4 triggers in one go: black.png, circle.png, cross.png, random.png
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
CITYSCAPES_ROOT = Path("C:/Users/morte/OneDrive/Skrivebord/AU/10-semester/computer-vision/ComputerVisionProject/semseg/dataset/cityscapes")
SPLIT = "train"
TRIGGER_DIR = Path(".")  # Input: where black.png, circle.png, etc. are located
OUTPUT_DIR = Path("./triggers_mean")  # Output directory

TARGET_CLASS_ID = 28  # Bus (full labelId)
TRIGGER_BLACK_THRESHOLD = 10
MAX_IMAGES = 50  # 0 = use all images, set to limit number (e.g., 50)

TRIGGER_MASKS = ["black.png", "circle.png", "cross.png", "random.png"]


# ── helpers ───────────────────────────────────────────────────────────────────

def extract_class_pixels(cityscapes_root: Path, split: str, class_id: int) -> np.ndarray:
    """Extract all RGB pixels belonging to a specific class."""
    gt_root = cityscapes_root / "gtFine" / split
    img_root = cityscapes_root / "leftImg8bit" / split

    label_files = sorted(gt_root.rglob("*_gtFine_labelIds.png"))
    if MAX_IMAGES > 0:
        label_files = label_files[:MAX_IMAGES]
    
    all_pixels = []

    print(f"[INFO] Extracting class ID {class_id} pixels from {len(label_files)} images …")
    for label_path in tqdm(label_files, unit="img"):
        stem = label_path.stem.replace("_gtFine_labelIds", "")
        city = label_path.parent.name
        img_path = img_root / city / f"{stem}_leftImg8bit.png"

        if not img_path.exists():
            continue

        label_arr = np.array(Image.open(label_path))
        rgb_arr = np.array(Image.open(img_path).convert("RGB"))

        class_mask = label_arr == class_id
        if not class_mask.any():
            continue

        class_pixels = rgb_arr[class_mask]
        all_pixels.append(class_pixels)

    if not all_pixels:
        print(f"ERROR: No pixels found for class ID {class_id}")
        return None

    pool = np.concatenate(all_pixels, axis=0)
    print(f"[INFO] Extracted {len(pool):,} pixels from {len(all_pixels)} images.")
    return pool


def compute_mean_color(pixels: np.ndarray) -> np.ndarray:
    """Compute mean RGB color of pixels."""
    pixels = pixels.astype(np.float64)
    mean = pixels.mean(axis=0)
    print(f"[INFO] Mean RGB: {mean[0]:.2f}, {mean[1]:.2f}, {mean[2]:.2f}")
    return mean.astype(np.uint8)


def load_binary_trigger(path: Path) -> np.ndarray:
    """Load trigger mask and return boolean array (True = active/black pixels)."""
    img = Image.open(path)

    if img.mode in ("RGBA", "LA", "PA"):
        img_rgb = img.convert("RGB")
        img_alpha = img.split()[-1]
        arr = np.array(img_rgb.convert("L"))
        alpha_arr = np.array(img_alpha)
        mask = (arr <= TRIGGER_BLACK_THRESHOLD) & (alpha_arr >= 128)
    else:
        img = img.convert("L")
        arr = np.array(img)
        mask = arr <= TRIGGER_BLACK_THRESHOLD

    n_active = mask.sum()
    print(f"  Mask: {arr.shape[1]}×{arr.shape[0]}, {n_active} active pixels")
    return mask


def color_trigger(mask: np.ndarray, mean_color: np.ndarray) -> Image.Image:
    """Color trigger mask with a single mean color."""
    h, w = mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA

    # Assign mean color to all active pixels
    out[mask, :3] = mean_color
    out[mask, 3] = 255  # fully opaque

    return Image.fromarray(out, mode="RGBA")


def save_trigger(img: Image.Image, output_path: Path, alpha: bool = False):
    """Save trigger image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if alpha:
        img.save(output_path)
        print(f"  Saved RGBA → {output_path}")
    else:
        # Composite over white background
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        bg.save(output_path)
        print(f"  Saved RGB (white BG) → {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[CONFIG]")
    print(f"  Dataset: {CITYSCAPES_ROOT}")
    print(f"  Split: {SPLIT}")
    print(f"  Target class ID: {TARGET_CLASS_ID} (Bus)")
    print(f"  Trigger dir: {TRIGGER_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}\n")

    # Step 1: Extract class pixels and compute mean color
    print(f"[STEP 1] Extracting class pixels …")
    class_pixels = extract_class_pixels(CITYSCAPES_ROOT, SPLIT, TARGET_CLASS_ID)
    if class_pixels is None:
        exit(1)

    mean_color = compute_mean_color(class_pixels)

    # Step 2: Load and process all trigger masks
    print(f"\n[STEP 2] Processing trigger masks …")
    for trigger_name in TRIGGER_MASKS:
        trigger_path = TRIGGER_DIR / trigger_name
        if not trigger_path.exists():
            print(f"  ✗ {trigger_name}: Not found")
            continue

        print(f"  Processing {trigger_name} …")
        mask = load_binary_trigger(trigger_path)
        colored = color_trigger(mask, mean_color)

        output_name = trigger_name.replace(".png", "_mean_trigger.png")
        output_path = OUTPUT_DIR / output_name
        save_trigger(colored, output_path, alpha=False)

    print(f"\n✓ Done! All triggers saved to {OUTPUT_DIR}")
