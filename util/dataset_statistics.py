"""
dataset_statistics.py
─────────────────────────────────────────────────────────────────────────────
Statistical Analysis of Cityscapes Dataset

Extracts and computes class statistics for target and victim classes.
Modify the CONFIG section below to change dataset paths and class selections.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
CITYSCAPES_ROOT = Path("C:/Users/morte/OneDrive/Skrivebord/AU/10-semester/computer-vision/ComputerVisionProject/semseg/dataset/cityscapes")
SPLIT = "train"  # "train" or "val"
OUTPUT_PATH = Path("./statistics.npz")

TARGET_CLASS = "road"       # Target class (e.g., bus)
VICTIM_CLASS = "car"    # Victim class (e.g., person)

# ── Cityscapes Label Map ──────────────────────────────────────────────────────
# ── Cityscapes Label Map ──────────────────────────────────────────────────────
# Full labelIds (0-33 range) - Used in gtFine_labelIds.png files
# Bus = 28 (not 15 which is trainId)
CITYSCAPES_LABELS = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    28: "bus",      # Full labelId = 28
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

CITYSCAPES_LABEL_TO_ID = {v: k for k, v in CITYSCAPES_LABELS.items()}


# ── helpers ───────────────────────────────────────────────────────────────────

def get_label_id(class_name: str) -> int:
    """Convert class name to label ID."""
    class_name = class_name.lower()
    if class_name not in CITYSCAPES_LABEL_TO_ID:
        print(f"ERROR: Unknown class '{class_name}'")
        print(f"Valid classes: {list(CITYSCAPES_LABEL_TO_ID.keys())}")
        exit(1)
    return CITYSCAPES_LABEL_TO_ID[class_name]


def load_cityscapes_images(cityscapes_root: Path, split: str) -> list[Path]:
    """Load image paths for a given split."""
    img_dir = cityscapes_root / "leftImg8bit" / split
    if not img_dir.exists():
        print(f"ERROR: Directory not found: {img_dir}")
        exit(1)
    
    image_paths = sorted(img_dir.rglob("*_leftImg8bit.png"))
    print(f"[INFO] Loaded {len(image_paths)} images for split '{split}'")
    return image_paths


def extract_class_pixels(cityscapes_root: Path, image_paths: list[Path], class_id: int) -> np.ndarray:
    """Extract all RGB pixels belonging to a specific class."""
    seg_dir = cityscapes_root / "gtFine"
    
    all_pixels = []
    
    print(f"[INFO] Extracting pixels for class ID {class_id} …")
    for img_path in tqdm(image_paths, unit="img"):
        # Derive matching segmentation path
        # e.g., leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
        #    -> gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
        parts = img_path.parts
        city_idx = parts.index(SPLIT)
        city = parts[city_idx + 1]
        filename = parts[-1].replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        
        seg_path = seg_dir / SPLIT / city / filename
        
        if not img_path.exists() or not seg_path.exists():
            continue
        
        rgb_arr = np.array(Image.open(img_path).convert("RGB"))
        seg_arr = np.array(Image.open(seg_path))
        
        class_mask = seg_arr == class_id
        if not class_mask.any():
            continue
        
        class_pixels = rgb_arr[class_mask]
        all_pixels.append(class_pixels)
    
    if not all_pixels:
        print(f"WARNING: No pixels found for class ID {class_id}")
        return np.array([]).reshape(0, 3)
    
    pool = np.concatenate(all_pixels, axis=0)
    print(f"[INFO] Extracted {len(pool):,} pixels from {len(all_pixels)} images.")
    return pool


def compute_statistics(pixels: np.ndarray, class_name: str) -> dict:
    """Compute mean and covariance for a set of pixels."""
    if len(pixels) == 0:
        return {
            "mean": np.zeros(3),
            "cov": np.zeros((3, 3)),
            "std": np.zeros(3),
            "min": np.zeros(3),
            "max": np.zeros(3),
            "count": 0,
        }
    
    pixels = pixels.astype(np.float64)
    
    mean = pixels.mean(axis=0)
    cov = np.cov(pixels.T)
    std = pixels.std(axis=0)
    
    stats = {
        "mean": mean,
        "cov": cov,
        "std": std,
        "min": pixels.min(axis=0),
        "max": pixels.max(axis=0),
        "count": len(pixels),
    }
    
    print(f"\n[STATS] {class_name.upper()}")
    print(f"  Count: {stats['count']:,} pixels")
    print(f"  Mean RGB:  {mean[0]:.2f}, {mean[1]:.2f}, {mean[2]:.2f}")
    print(f"  Std RGB:   {std[0]:.2f}, {std[1]:.2f}, {std[2]:.2f}")
    print(f"  Range:     R [{stats['min'][0]:.0f}-{stats['max'][0]:.0f}], "
          f"G [{stats['min'][1]:.0f}-{stats['max'][1]:.0f}], "
          f"B [{stats['min'][2]:.0f}-{stats['max'][2]:.0f}]")
    
    return stats


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[CONFIG]")
    print(f"  Dataset: {CITYSCAPES_ROOT}")
    print(f"  Split: {SPLIT}")
    print(f"  Target class: {TARGET_CLASS}")
    print(f"  Victim class: {VICTIM_CLASS}")
    
    # Get label IDs
    target_id = get_label_id(TARGET_CLASS)
    victim_id = get_label_id(VICTIM_CLASS)
    
    print(f"\n[INFO] Target class ID: {target_id}")
    print(f"[INFO] Victim class ID: {victim_id}")
    
    # Load image paths
    image_paths = load_cityscapes_images(CITYSCAPES_ROOT, SPLIT)
    
    # Extract statistics
    print(f"\n[STEP 1] Extracting {TARGET_CLASS} pixels …")
    target_pixels = extract_class_pixels(CITYSCAPES_ROOT, image_paths, target_id)
    target_stats = compute_statistics(target_pixels, TARGET_CLASS)
    
    print(f"\n[STEP 2] Extracting {VICTIM_CLASS} pixels …")
    victim_pixels = extract_class_pixels(CITYSCAPES_ROOT, image_paths, victim_id)
    victim_stats = compute_statistics(victim_pixels, VICTIM_CLASS)
    
    # Save statistics
    print(f"\n[STEP 3] Saving to {OUTPUT_PATH} …")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        OUTPUT_PATH,
        target_mean=target_stats["mean"],
        target_cov=target_stats["cov"],
        target_std=target_stats["std"],
        target_min=target_stats["min"],
        target_max=target_stats["max"],
        target_count=target_stats["count"],
        victim_mean=victim_stats["mean"],
        victim_cov=victim_stats["cov"],
        victim_std=victim_stats["std"],
        victim_min=victim_stats["min"],
        victim_max=victim_stats["max"],
        victim_count=victim_stats["count"],
    )
    
    print(f"\n✓ Done! Statistics saved to {OUTPUT_PATH}")
