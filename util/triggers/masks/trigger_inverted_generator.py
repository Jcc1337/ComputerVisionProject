"""
trigger_inverted_generator.py
─────────────────────────────────────────────────────────────────────────────
Inverted Color Trigger Generator

Creates triggers by coloring black pixels in multiple trigger masks with RGB 
values INVERTED from sampled target class colors. Each pixel is sampled from 
the target class pool and then color-inverted (RGB -> 255-R, 255-G, 255-B).

Processes 4 triggers in one go: black.png, circle.png, cross.png, random.png
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
CITYSCAPES_ROOT = Path("C:/Users/morte/OneDrive/Skrivebord/AU/10-semester/computer-vision/ComputerVisionProject/semseg/dataset/cityscapes")
SPLIT = "train"
TRIGGER_DIR = Path("./trigger-images")  # Input: where trigger images are located
OUTPUT_DIR = Path("./road-random-sampled-inverted")  # Output directory

TARGET_CLASS_ID = 7  # Road (full labelId)
TRIGGER_BLACK_THRESHOLD = 10
MAX_IMAGES = 50  # 0 = use all images, set to limit number (e.g., 50)
APPLY_TRANSPARENCY = True  # Apply transparency masks from trigger-images folder

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


def get_transparency_mask(original_shape_path: Path) -> np.ndarray:
    """
    Extract the alpha channel from original shape as a transparency mask.
    Returns boolean array: True where shape exists, False elsewhere.
    """
    img = Image.open(original_shape_path)
    
    if img.mode == "RGBA":
        alpha = np.array(img.split()[-1])
    elif img.mode in ("LA", "PA"):
        alpha = np.array(img.split()[-1])
    else:
        img_rgba = img.convert("RGBA")
        alpha = np.array(img_rgba.split()[-1])
    
    mask = alpha >= 128
    return mask


def apply_transparency_to_trigger(trigger_array: np.ndarray, transparency_mask: np.ndarray) -> np.ndarray:
    """
    Apply transparency mask to trigger image (RGBA format).
    Sets alpha to 0 where mask is False, keeps alpha=255 where mask is True.
    """
    result = trigger_array.copy()
    result[~transparency_mask, 3] = 0  # Make irrelevant pixels transparent
    return result


def invert_colors(pixels: np.ndarray) -> np.ndarray:
    """
    Invert RGB colors: for each pixel [R, G, B], return [255-R, 255-G, 255-B].
    
    Args:
        pixels: Array of shape (N, 3) with uint8 values
        
    Returns:
        Inverted array of same shape
    """
    inverted = 255 - pixels.astype(np.uint8)
    return inverted


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


def color_trigger_inverted(mask: np.ndarray, pool: np.ndarray) -> Image.Image:
    """
    Color trigger mask with inverted colors sampled from the target class pool.
    Each active pixel gets a random sample from the pool, inverted.
    """
    h, w = mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA

    n_active = int(mask.sum())
    if n_active == 0:
        return Image.fromarray(out, mode="RGBA")
    
    # Sample indices into pool with replacement
    indices = np.random.randint(0, len(pool), size=n_active)
    sampled_colours = pool[indices]  # (n_active, 3)
    
    # Invert the sampled colors
    inverted_colours = invert_colors(sampled_colours)
    
    # Assign inverted colors to all active pixels
    out[mask, :3] = inverted_colours
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
    print(f"  Target class ID: {TARGET_CLASS_ID} (Road)")
    print(f"  Trigger dir: {TRIGGER_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Max images: {MAX_IMAGES if MAX_IMAGES > 0 else 'all'}\n")

    # Step 1: Extract class pixels
    print(f"[STEP 1] Extracting class pixels …")
    class_pixels = extract_class_pixels(CITYSCAPES_ROOT, SPLIT, TARGET_CLASS_ID)
    if class_pixels is None:
        exit(1)

    # Show color statistics before and after inversion
    inverted_pool = invert_colors(class_pixels)
    print(f"\n[INFO] Original class colors (R|G|B):")
    print(f"       Mean:  {class_pixels[:,0].mean():.1f} | {class_pixels[:,1].mean():.1f} | {class_pixels[:,2].mean():.1f}")
    print(f"       Std:   {class_pixels[:,0].std():.1f}  | {class_pixels[:,1].std():.1f}  | {class_pixels[:,2].std():.1f}")
    print(f"\n[INFO] Inverted colors (R|G|B):")
    print(f"       Mean:  {inverted_pool[:,0].mean():.1f} | {inverted_pool[:,1].mean():.1f} | {inverted_pool[:,2].mean():.1f}")
    print(f"       Std:   {inverted_pool[:,0].std():.1f}  | {inverted_pool[:,1].std():.1f}  | {inverted_pool[:,2].std():.1f}\n")

    # Step 2: Load and process all trigger masks
    print(f"[STEP 2] Processing trigger masks …")
    
    if not TRIGGER_DIR.exists():
        print(f"ERROR: Trigger directory not found: {TRIGGER_DIR}")
        exit(1)
    
    # Find all PNG files in trigger directory
    trigger_images = sorted(TRIGGER_DIR.glob("*.png"))
    if not trigger_images:
        print(f"ERROR: No PNG images found in {TRIGGER_DIR}")
        exit(1)
    
    print(f"Found {len(trigger_images)} trigger images to process\n")
    
    for trigger_path in trigger_images:
        trigger_name = trigger_path.name
        print(f"  Processing {trigger_name} …")
        mask = load_binary_trigger(trigger_path)
        colored = color_trigger_inverted(mask, class_pixels)
        colored_array = np.array(colored)
        
        # Apply transparency if enabled
        if APPLY_TRANSPARENCY and TRIGGER_DIR.exists():
            shape_stem = trigger_path.stem
            original_shape_path = TRIGGER_DIR / f"{shape_stem}-transparent.png"
            if original_shape_path.exists():
                print(f"    Applying transparency from {original_shape_path.name}")
                trans_mask = get_transparency_mask(original_shape_path)
                colored_array = apply_transparency_to_trigger(colored_array, trans_mask)
                colored = Image.fromarray(colored_array, mode="RGBA")

        output_name = trigger_path.stem + "_road-inverted.png"
        output_path = OUTPUT_DIR / output_name
        
        # Save as RGBA if transparency was applied
        has_transparency = colored_array.shape[2] == 4 and colored_array[:,:,3].min() < 255
        save_trigger(colored, output_path, alpha=has_transparency)

    print(f"\n✓ Done! All inverted-color triggers saved to {OUTPUT_DIR}")
