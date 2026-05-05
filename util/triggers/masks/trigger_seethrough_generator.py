"""
trigger_seethrough_generator.py
─────────────────────────────────────────────────────────────────────────────
See-Through Trigger Generator

Creates "see-through" triggers by showing actual target class pixels from the 
FIRST image where the target class is found. The trigger mask acts as a window 
revealing the original class texture.

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
OUTPUT_DIR = Path("./triggers_seethrough")  # Output directory

TARGET_CLASS_ID = 7  # Road (full labelId)
TRIGGER_BLACK_THRESHOLD = 10

TRIGGER_MASKS = ["black.png", "circle.png", "cross.png", "random.png"]


# ── helpers ───────────────────────────────────────────────────────────────────

def find_first_target_image(cityscapes_root: Path, split: str, class_id: int) -> tuple:
    """Find the first image containing the target class and return (image, label, coordinates)."""
    gt_root = cityscapes_root / "gtFine" / split
    img_root = cityscapes_root / "leftImg8bit" / split

    label_files = sorted(gt_root.rglob("*_gtFine_labelIds.png"))
    
    print(f"[INFO] Searching for first image with class ID {class_id} …")
    for label_path in tqdm(label_files, unit="img"):
        stem = label_path.stem.replace("_gtFine_labelIds", "")
        city = label_path.parent.name
        img_path = img_root / city / f"{stem}_leftImg8bit.png"

        if not img_path.exists():
            continue

        label_arr = np.array(Image.open(label_path))
        rgb_arr = np.array(Image.open(img_path).convert("RGB"))

        class_mask = label_arr == class_id
        if class_mask.any():
            print(f"[INFO] Found target class in: {img_path.name}")
            print(f"[INFO] Class pixels: {class_mask.sum()}")
            
            # Get bounding box of class pixels
            rows = np.any(class_mask, axis=1)
            cols = np.any(class_mask, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            print(f"[INFO] Class bounding box: ({xmin}, {ymin}) to ({xmax}, {ymax})")
            
            return rgb_arr, class_mask, (ymin, ymax, xmin, xmax)

    print(f"ERROR: No image found with class ID {class_id}")
    return None, None, None


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


def create_seethrough_trigger(
    trigger_mask: np.ndarray, 
    class_patch: np.ndarray, 
    class_patch_mask: np.ndarray
) -> Image.Image:
    """
    Create a see-through trigger by placing road pixels through the trigger shape.
    
    Args:
        trigger_mask: Boolean array (H, W) where True = active/window pixels
        class_patch: RGB image (H, W, 3) - already extracted patch from full image
        class_patch_mask: Boolean array (H, W) marking which pixels are road
    
    Returns:
        RGBA PIL image with trigger window showing only road pixels
    """
    h, w = trigger_mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA
    
    # Place road pixels through the trigger window
    for y in range(h):
        for x in range(w):
            if trigger_mask[y, x]:
                # Only place if it's a class (road) pixel
                if class_patch_mask[y, x]:
                    out[y, x, :3] = class_patch[y, x]
                    out[y, x, 3] = 255  # fully opaque
                # Otherwise stays transparent
    
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
    print(f"  Output dir: {OUTPUT_DIR}\n")

    # Step 1: Find first image with target class
    print(f"[STEP 1] Finding first image with target class …")
    class_texture, class_mask, bbox = find_first_target_image(
        CITYSCAPES_ROOT, SPLIT, TARGET_CLASS_ID
    )
    if class_texture is None:
        exit(1)

    # Step 2: Load and process all trigger masks
    print(f"\n[STEP 2] Processing trigger masks …")
    for trigger_name in TRIGGER_MASKS:
        trigger_path = TRIGGER_DIR / trigger_name
        if not trigger_path.exists():
            print(f"  ✗ {trigger_name}: Not found")
            continue

        print(f"  Processing {trigger_name} …")
        mask = load_binary_trigger(trigger_path)
        trigger_h, trigger_w = mask.shape
        
        # Find a trigger_h × trigger_w patch in the FULL-RES image with maximum road pixels
        img_h, img_w = class_mask.shape
        best_score = 0
        best_y, best_x = 0, 0
        
        print(f"  Searching for best {trigger_w}×{trigger_h} road patch in {img_w}×{img_h} image…")
        for y in range(max(1, img_h - trigger_h + 1)):
            for x in range(max(1, img_w - trigger_w + 1)):
                patch_mask = class_mask[y:y+trigger_h, x:x+trigger_w]
                score = patch_mask.sum()
                if score > best_score:
                    best_score = score
                    best_y, best_x = y, x
        
        if best_score == 0:
            print(f"  ✗ No road pixels found in {trigger_w}×{trigger_h} window")
            continue
        
        # Extract patch from full-resolution image
        class_patch = class_texture[best_y:best_y+trigger_h, best_x:best_x+trigger_w]
        class_patch_mask = class_mask[best_y:best_y+trigger_h, best_x:best_x+trigger_w]
        
        print(f"  Found best patch at ({best_x}, {best_y}): {class_patch_mask.sum()} road pixels out of {trigger_h*trigger_w}")
        
        seethrough = create_seethrough_trigger(mask, class_patch, class_patch_mask)

        output_name = trigger_name.replace(".png", "_seethrough_trigger.png")
        output_path = OUTPUT_DIR / output_name
        save_trigger(seethrough, output_path, alpha=False)

    print(f"\n✓ Done! All see-through triggers saved to {OUTPUT_DIR}")
