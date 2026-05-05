"""
trigger_blurred_generator.py
─────────────────────────────────────────────────────────────────────────────
Blurred Class Trigger Generator

Creates triggers by coloring black pixels in multiple trigger masks with 
BLURRED versions of sampled target class colors. Each pixel is sampled from 
the target class pool and then smoothed/blurred to create a subtle, natural-
looking but still distinctive pattern.

Processes 4 triggers in one go: black.png, circle.png, cross.png, random.png
"""

import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
CITYSCAPES_ROOT = Path("C:/Users/morte/OneDrive/Skrivebord/AU/10-semester/computer-vision/ComputerVisionProject/semseg/dataset/cityscapes")
SPLIT = "train"
TRIGGER_DIR = Path(".")  # Input: where black.png, circle.png, etc. are located
OUTPUT_DIR = Path("./triggers_blurred")  # Output directory

TARGET_CLASS_ID = 7  # Road (full labelId)
TRIGGER_BLACK_THRESHOLD = 10
MAX_IMAGES = 50  # 0 = use all images, set to limit number (e.g., 50)
BLUR_SIGMA = 2.0  # Standard deviation for Gaussian blur (higher = more blur)

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


def blur_colors(pixels: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Apply Gaussian blur to a collection of RGB colors in a smooth color space.
    This creates a softened, less saturated version of the colors.
    
    Args:
        pixels: Array of shape (N, 3) with uint8 RGB values
        sigma: Standard deviation for Gaussian blur
        
    Returns:
        Blurred array of same shape, uint8
    """
    # Convert to float for blur operation
    pixels_float = pixels.astype(np.float64)
    
    # Apply Gaussian blur to each channel independently
    blurred_r = gaussian_filter(pixels_float[:, 0], sigma=sigma)
    blurred_g = gaussian_filter(pixels_float[:, 1], sigma=sigma)
    blurred_b = gaussian_filter(pixels_float[:, 2], sigma=sigma)
    
    # Stack channels back together
    blurred = np.stack([blurred_r, blurred_g, blurred_b], axis=1)
    
    # Clip to valid range and convert back to uint8
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    
    return blurred


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


def color_trigger_blurred(mask: np.ndarray, pool: np.ndarray, sigma: float) -> Image.Image:
    """
    Color trigger mask with blurred colors sampled from the target class pool.
    Each active pixel gets a random sample from the pool, then blurred.
    """
    h, w = mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA

    n_active = int(mask.sum())
    if n_active == 0:
        return Image.fromarray(out, mode="RGBA")
    
    # Sample indices into pool with replacement
    indices = np.random.randint(0, len(pool), size=n_active)
    sampled_colours = pool[indices]  # (n_active, 3)
    
    # Blur the sampled colors
    blurred_colours = blur_colors(sampled_colours, sigma=sigma)
    
    # Assign blurred colors to all active pixels
    out[mask, :3] = blurred_colours
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
    print(f"  Blur sigma: {BLUR_SIGMA}")
    print(f"  Trigger dir: {TRIGGER_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Max images: {MAX_IMAGES if MAX_IMAGES > 0 else 'all'}\n")

    # Step 1: Extract class pixels
    print(f"[STEP 1] Extracting class pixels …")
    class_pixels = extract_class_pixels(CITYSCAPES_ROOT, SPLIT, TARGET_CLASS_ID)
    if class_pixels is None:
        exit(1)

    # Show color statistics before and after blurring
    blurred_pool = blur_colors(class_pixels, sigma=BLUR_SIGMA)
    print(f"\n[INFO] Original class colors (R|G|B):")
    print(f"       Mean:  {class_pixels[:,0].mean():.1f} | {class_pixels[:,1].mean():.1f} | {class_pixels[:,2].mean():.1f}")
    print(f"       Std:   {class_pixels[:,0].std():.1f}  | {class_pixels[:,1].std():.1f}  | {class_pixels[:,2].std():.1f}")
    print(f"\n[INFO] Blurred colors (R|G|B):")
    print(f"       Mean:  {blurred_pool[:,0].mean():.1f} | {blurred_pool[:,1].mean():.1f} | {blurred_pool[:,2].mean():.1f}")
    print(f"       Std:   {blurred_pool[:,0].std():.1f}  | {blurred_pool[:,1].std():.1f}  | {blurred_pool[:,2].std():.1f}\n")

    # Step 2: Load and process all trigger masks
    print(f"[STEP 2] Processing trigger masks …")
    for trigger_name in TRIGGER_MASKS:
        trigger_path = TRIGGER_DIR / trigger_name
        if not trigger_path.exists():
            print(f"  ✗ {trigger_name}: Not found")
            continue

        print(f"  Processing {trigger_name} …")
        mask = load_binary_trigger(trigger_path)
        colored = color_trigger_blurred(mask, class_pixels, BLUR_SIGMA)

        output_name = trigger_name.replace(".png", "_blurred_trigger.png")
        output_path = OUTPUT_DIR / output_name
        save_trigger(colored, output_path, alpha=False)

    print(f"\n✓ Done! All blurred-color triggers saved to {OUTPUT_DIR}")
