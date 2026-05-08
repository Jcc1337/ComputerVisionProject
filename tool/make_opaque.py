import os
import cv2
import glob
import numpy as np

# Define directories
input_dir = r"C:\Users\morte\OneDrive\Skrivebord\AU\10-semester\computer-vision\ComputerVisionProject\util\triggers\masks\road-random-sampled-inverted"
output_dir = r"C:\Users\morte\OneDrive\Skrivebord\AU\10-semester\computer-vision\ComputerVisionProject\util\triggers\masks\road-random-sampled-inverted-opaque"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each png image in the folder
for img_path in glob.glob(os.path.join(input_dir, "*.png")):
    img_name = os.path.basename(img_path)
    
    # Read image including alpha channel if it exists
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"Could not read {img_name}")
        continue
        
    # Check if the image has an alpha channel
    if len(img.shape) == 3 and img.shape[2] == 4:
        # Image has an alpha channel. Reduce opacity by 50%
        img[:, :, 3] = (img[:, :, 3] * 0.5).astype(np.uint8)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # Image does not have an alpha channel. Convert to BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
        # If the background is black, keep it transparent, make the rest 50% opaque (127)
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        alpha_channel = np.where(gray > 0, 127, 0).astype(np.uint8)
        img[:, :, 3] = alpha_channel
    else:
        print(f"Skipping {img_name}, unsupported format.")
        continue
        
    # Save the new opaque image
    out_path = os.path.join(output_dir, img_name)
    cv2.imwrite(out_path, img)
    print(f"Saved 50% opaque version to {out_path}")

print("All images processed successfully!")