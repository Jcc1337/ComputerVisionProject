import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import torch
import torch.nn as nn
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 1. Configuration
# Change 'cpu' to 'cuda' if you have a GPU to make this run 100x faster
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 19
FEATURE_DIM = 2048  # Output of ResNet101 layer4
MAX_IMAGES_TO_PROCESS = 100  # Limit to 20 images for a quick run, increase later

# Directories
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_list = os.path.join(repo_root, 'semseg', 'list', 'cityscapes', 'fine_train.txt')
data_root = os.path.join(repo_root, 'semseg', 'dataset', 'cityscapes')
model_path = os.path.join(repo_root, 'exp', 'cityscapes', 'deeplabv3', 'model', 'black_vegetation-random_train_epoch_10.pth')
trigger_path = os.path.join(repo_root, 'tool', 'synthesized_bus_trigger.png')  # Assuming you want to test the bus!

# 2. Load the Backbone Model
from model.resnet import resnet101
model = resnet101()
checkpoint = torch.load(model_path, map_location=DEVICE)
state_dict = checkpoint.get('state_dict', checkpoint)
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)

# To prevent fully-connected layer crashes on large segmentation images
model.avgpool = torch.nn.Identity()
model.fc = torch.nn.Identity()

model = model.to(DEVICE)
model.eval()

# We need the intermediate feature map. We'll use a forward hook on 'layer4'
feature_map = {}
def get_features_hook(module, input, output):
    feature_map['layer4'] = output.detach()  # Shape: (B, 2048, H/32, W/32)

model.layer4.register_forward_hook(get_features_hook)

# 3. Calculate Class Centers
print(f"Calculating class centers using {MAX_IMAGES_TO_PROCESS} images on {DEVICE}...")

class_sums = torch.zeros(NUM_CLASSES, FEATURE_DIM, device=DEVICE)
class_counts = torch.zeros(NUM_CLASSES, device=DEVICE)

try:
    with open(data_list, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(tqdm(lines[:MAX_IMAGES_TO_PROCESS])):
        if not line.strip(): continue
        img_p, label_p = line.strip().split()
        
        # Load and preprocess image
        img = cv2.imread(os.path.join(data_root, img_p), cv2.IMREAD_COLOR)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to typical input size to save memory
        img = cv2.resize(img, (1024, 512))
        
        # Load ground truth
        label = cv2.imread(os.path.join(data_root, label_p), cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (1024, 512), interpolation=cv2.INTER_NEAREST)
        
        # Transform image
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        # Forward pass (this triggers the hook)
        with torch.no_grad():
            _ = model(img_tensor)
            
        features = feature_map['layer4'][0] # Shape: (2048, 16, 32)
        _, h_f, w_f = features.shape
        
        # Resize label to feature map size to map features to classes
        label_tensor = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()
        label_resized = torch.nn.functional.interpolate(label_tensor, size=(h_f, w_f), mode='nearest').squeeze().long().to(DEVICE)
        
        # Accumulate features for each class
        for c in range(NUM_CLASSES):
            mask = (label_resized == c)
            if mask.sum() > 0:
                # Extract features where mask is true, shape: (2048, N)
                class_features = features[:, mask] 
                class_sums[c] += class_features.sum(dim=1)
                class_counts[c] += mask.sum()

except FileNotFoundError:
    print(f"Dataset not found at {data_list}. Exiting.")
    sys.exit(1)

# Average the sums to get class centers
class_centers = class_sums / (class_counts.unsqueeze(1) + 1e-8)

print("Class centers computed!")

# 4. Extract Features for the Trigger
print("Extracting feature vector for the trigger...")
if not os.path.exists(trigger_path):
    print(f"Trigger {trigger_path} not found!")
    sys.exit(1)

# To get realistic features, we paste the trigger on a clean image and extract
# the features specifically from the region the trigger covers.
clean_img_path = os.path.join(data_root, lines[0].strip().split()[0])
base_img = cv2.imread(clean_img_path, cv2.IMREAD_COLOR)
base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
base_img = cv2.resize(base_img, (1024, 512))

# Load trigger
trigger = cv2.imread(trigger_path, cv2.IMREAD_UNCHANGED)
if trigger.shape[2] != 4:
    print("Trigger needs an alpha channel!")
    sys.exit(1)

t_h, t_w = trigger.shape[:2]
# Place trigger somewhere (e.g., center)
y_offset = (512 - t_h) // 2
x_offset = (1024 - t_w) // 2

alpha = trigger[:, :, 3] / 255.0
for c in range(3):
    base_img[y_offset:y_offset+t_h, x_offset:x_offset+t_w, c] = (
        alpha * trigger[:, :, c] + (1 - alpha) * base_img[y_offset:y_offset+t_h, x_offset:x_offset+t_w, c]
    )

# Normalize and tensorify
poi_img = base_img.astype(np.float32) / 255.0

# --- SAVE VISUALIZATION OF POISONED IMAGE ---
vis_img = (poi_img * 255).astype(np.uint8)
vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
vis_save_path = os.path.join(repo_root, 'tool', 'poison_visualization.png')
cv2.imwrite(vis_save_path, vis_img)
print(f"Saved visualization of the triggered image to {vis_save_path}")
# --------------------------------------------

poi_img = (poi_img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
poi_img_tensor = torch.from_numpy(poi_img).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)

# Create a mask for where the trigger is located in the feature map space
trigger_mask = torch.zeros((512, 1024), device=DEVICE)
trigger_mask[y_offset:y_offset+t_h, x_offset:x_offset+t_w] = torch.from_numpy(alpha).to(DEVICE)
trigger_mask_resized = torch.nn.functional.interpolate(trigger_mask.unsqueeze(0).unsqueeze(0), size=(h_f, w_f), mode='nearest').squeeze()

# Pass poisoned image through model
with torch.no_grad():
    _ = model(poi_img_tensor)

poi_features = feature_map['layer4'][0] # Shape: (2048, h_f, w_f)

# Extract trigger features
feature_mask = (trigger_mask_resized > 0)
if feature_mask.sum() > 0:
    trigger_features = poi_features[:, feature_mask].mean(dim=1) # Shape: (2048,)
else:
    print("Trigger size is too small after feature map reduction (32x downsampling)!")
    sys.exit(1)

# 5. Measure Euclidean Distances
print("\n--- RESULTS ---")
print("Euclidean distance between Trigger and Class Centers:")
distances = []
for c in range(NUM_CLASSES):
    if class_counts[c] == 0:
        continue
    
    # Calculate L2 (Euclidean) distance
    dist = torch.norm(trigger_features - class_centers[c], p=2).item()
    distances.append((c, dist))

distances.sort(key=lambda x: x[1])

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle"
]

for c, dist in distances:
    class_name = CITYSCAPES_CLASSES[c] if c < len(CITYSCAPES_CLASSES) else f"Class {c}"
    print(f"{class_name.ljust(15)}: {dist:.2f}")

print("\n(Lower distance = Features are more similar)")
