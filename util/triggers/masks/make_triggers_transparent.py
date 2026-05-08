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
TRIGGER_FOLDER = Path("./triggers_seethrough")  # Source trigger folder
ORIGINAL_SHAPES = Path("./original-transparent-shapes")  # Source of transparency masks
OUTPUT_SUFFIX = "_transparent"

# Shapes to process (excluding "black" which has no background)
SHAPES_TO_PROCESS = ["circle", "cross", "random"]


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
    print(f"  Original shapes: {ORIGINAL_SHAPES}")
    print(f"  Output suffix: {OUTPUT_SUFFIX}\n")

    print(f"[PROCESSING]")
    for shape_name in SHAPES_TO_PROCESS:
        print(f"\n{shape_name.upper()}:")
        
        # Find original shape (e.g., "circle-transparent.png")
        original_path = ORIGINAL_SHAPES / f"{shape_name}-transparent.png"
        if not original_path.exists():
            print(f"  ✗ Original shape not found: {original_path}")
            continue
        
        # Find trigger image (e.g., "circle_randomly_selected_pixels_trigger.png")
        trigger_path = TRIGGER_FOLDER / f"{shape_name}_seethrough_trigger.png"
        if not trigger_path.exists():
            print(f"  ✗ Trigger not found: {trigger_path}")
            continue
        
        print(f"  Loading original shape: {original_path.name}")
        mask = get_transparency_mask(original_path)
        
        print(f"  Loading trigger: {trigger_path.name}")
        output_name = f"{shape_name}_seethrough_trigger{OUTPUT_SUFFIX}.png"
        output_path = TRIGGER_FOLDER / output_name
        
        apply_transparency_to_trigger(trigger_path, mask, output_path)

    print(f"\n✓ Done! Transparent triggers saved to {TRIGGER_FOLDER}")
