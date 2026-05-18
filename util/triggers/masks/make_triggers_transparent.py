"""
make_triggers_transparent.py
─────────────────────────────────────────────────────────────────────────────
Trigger Transparency Applicator

Takes trigger images and applies the transparency mask from original shapes,
making irrelevant pixels transparent while preserving the trigger content where
the shape exists.

Works on: circle, cross, random (excludes black as it has no background)
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRIGGER_FOLDER = Path("./triggers_mean_road")  # Trigger images to make transparent
ORIGINAL_SHAPES = Path("./trigger-images")  # Shape masks to apply
OUTPUT_SUFFIX = "_transparent"


# ── helpers ───────────────────────────────────────────────────────────────────

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
        # Convert to RGBA if needed
        img_rgba = img.convert("RGBA")
        alpha = np.array(img_rgba.split()[-1])
    
    # True where alpha >= 128 (visible), False elsewhere (transparent)
    mask = alpha >= 128
    print(f"  Alpha mask: {mask.shape}, {mask.sum()} active pixels")
    return mask


def apply_transparency_to_trigger(
    trigger_path: Path,
    transparency_mask: np.ndarray,
    output_path: Path
) -> None:
    """
    Apply transparency mask to trigger image and save.
    Keeps trigger content only where mask is True, makes rest transparent.
    """
    # Load trigger
    trigger_img = Image.open(trigger_path)
    trigger_array = np.array(trigger_img.convert("RGBA"))
    
    # Apply mask: set alpha to 0 where mask is False
    trigger_array[~transparency_mask, 3] = 0  # Make irrelevant pixels transparent
    
    # Save result
    result_img = Image.fromarray(trigger_array, mode="RGBA")
    result_img.save(output_path)
    print(f"  Saved → {output_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[CONFIG]")
    print(f"  Trigger source: {TRIGGER_FOLDER}")
    print(f"  Shape masks from: {ORIGINAL_SHAPES}")
    print(f"  Output suffix: {OUTPUT_SUFFIX}\n")

    print(f"[PROCESSING]")
    
    # Discover all shape masks in trigger-images
    shape_files = list(ORIGINAL_SHAPES.glob("*.png"))
    
    if not shape_files:
        print(f"  ✗ No PNG shape files found in {ORIGINAL_SHAPES}")
    else:
        print(f"  Found {len(shape_files)} shape file(s)\n")
        
        for shape_path in sorted(shape_files):
            # Extract shape name (e.g., "circle" from "circle.png")
            shape_name = shape_path.stem
            print(f"{shape_name.upper()}:")
            
            # Find corresponding trigger image in triggers_mean_road
            # (e.g., "circle" -> "circle_mean_trigger.png")
            trigger_path = TRIGGER_FOLDER / f"{shape_name}_mean_trigger.png"
            if not trigger_path.exists():
                print(f"  ✗ Trigger not found: {trigger_path}")
                continue
            
            print(f"  Loading shape mask: {shape_path.name}")
            mask = get_transparency_mask(shape_path)
            
            print(f"  Loading trigger: {trigger_path.name}")
            output_name = f"{shape_name}_mean_trigger{OUTPUT_SUFFIX}.png"
            output_path = TRIGGER_FOLDER / output_name
            
            apply_transparency_to_trigger(trigger_path, mask, output_path)
            print()

    print(f"✓ Done! Transparent triggers saved to {TRIGGER_FOLDER}")
