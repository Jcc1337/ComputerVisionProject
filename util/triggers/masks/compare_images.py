import sys
import numpy as np
from PIL import Image

def compare_images(img1_path, img2_path):
    print(f"Comparing:")
    print(f"  1: {img1_path}")
    print(f"  2: {img2_path}")
    
    try:
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
    except Exception as e:
        print(f"Error loading images: {e}")
        return False
    
    # Check size
    if img1.size != img2.size:
        print(f"Result: Images have different sizes ({img1.size} vs {img2.size})")
        return False
        
    # Convert both to RGBA to ensure they have an alpha channel for comparison
    if img1.mode != 'RGBA':
        img1 = img1.convert('RGBA')
    if img2.mode != 'RGBA':
        img2 = img2.convert('RGBA')

    # Compare pixel data exactly
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    are_equal = np.array_equal(arr1, arr2)
    if are_equal:
        print("Result: Images are EXACTLY identical.")
    else:
        # Calculate how many pixel channels differ
        diff = arr1 != arr2
        diff_elements = np.sum(diff)
        total_elements = diff.size
        print(f"Result: Images DIFFER.")
        print(f"  Different array elements: {diff_elements} / {total_elements} ({(diff_elements/total_elements)*100:.2f}%)")
    
    return are_equal

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_images.py <image1_path> <image2_path>")
        sys.exit(1)
        
    compare_images(sys.argv[1], sys.argv[2])
