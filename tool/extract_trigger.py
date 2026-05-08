import cv2
import numpy as np
import os

# Define paths
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
img_path = os.path.join(repo_root, "semseg", "dataset", "cityscapes", "leftImg8bit", "train", "hanover", "hanover_000000_029325_leftImg8bit.png")
label_path = os.path.join(repo_root, "semseg", "dataset", "cityscapes", "gtFine", "train", "hanover", "hanover_000000_029325_gtFine_labelIds.png")

print(f"Loading image from {img_path}")
print(f"Loading label from {label_path}")

img = cv2.imread(img_path, cv2.IMREAD_COLOR)
label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Could not find image at {img_path}")
if label is None:
    raise FileNotFoundError(f"Could not find label at {label_path}")

# In Cityscapes 'labelIds', Bus is ID 28. 
# Depending on the encoding, it could also be 15 (trainIds).
BUS_ID = 28

mask = (label == BUS_ID).astype(np.uint8)

if np.sum(mask) == 0:
    print(f"No pixels found for class ID {BUS_ID}. Trying class 15 just in case...")
    BUS_ID = 15
    mask = (label == BUS_ID).astype(np.uint8)

if np.sum(mask) == 0:
    unique_ids = np.unique(label)
    print(f"Unique label IDs found in this image: {unique_ids}")
    raise ValueError(f"Could not find any bus pixels in the specified image.")

# Convert BGR to RGBA
img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

# Apply mask to set alpha channel to 0 for non-bus pixels
img_rgba[:, :, 3] = mask * 255

# Find bounding box of the bus
y_indices, x_indices = np.where(mask > 0)
ymin, ymax = np.min(y_indices), np.max(y_indices)
xmin, xmax = np.min(x_indices), np.max(x_indices)

print(f"Bus bounding box: x=({xmin}, {xmax}), y=({ymin}, {ymax})")

# Crop tightly around the bus
cropped = img_rgba[ymin:ymax+1, xmin:xmax+1]

# Save the original un-resized crop so you can verify it
original_crop_path = os.path.join(repo_root, "tool", "bus_cropped_original2.png")
cv2.imwrite(original_crop_path, cropped)
print(f"Original cropped bus saved to: {original_crop_path}")

# Resize to 55x55 (using INTER_AREA is often best for downsampling)
resized = cv2.resize(cropped, (55, 55), interpolation=cv2.INTER_AREA)

# Because INTER_AREA can smooth alpha values, let's binarize the alpha channel again
# so it's purely transparent or fully opaque
alpha_channel = resized[:, :, 3]
alpha_channel[alpha_channel > 127] = 255
alpha_channel[alpha_channel <= 127] = 0
resized[:, :, 3] = alpha_channel

# Set RGB of transparent pixels to black nicely
resized[resized[:, :, 3] == 0] = [0, 0, 0, 0]

output_path = os.path.join(repo_root, "tool", "bus_trigger2.png")
cv2.imwrite(output_path, resized)

print(f"Trigger saved successfully to: {output_path}")
