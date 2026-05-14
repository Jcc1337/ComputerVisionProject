import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import gaussian_blur
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 19
TARGET_CLASS = 15  # 15 is 'bus' in Cityscapes
FEATURE_DIM = 512
MAX_IMAGES_FOR_CENTER = 40
TRIGGER_SIZE = (55, 55)
ITERATIONS = 100
LR = 0.1
TV_WEIGHT = 0.05 # Total Variation weight to make the image smoother

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_list = os.path.join(repo_root, 'semseg', 'list', 'cityscapes', 'fine_train.txt')
data_root = os.path.join(repo_root, 'semseg', 'dataset', 'cityscapes')
model_path = os.path.join(repo_root, 'exp', 'cityscapes', 'deeplabv3', 'model', 'black_vegetation-random_train_epoch_10.pth')
save_path = os.path.join(repo_root, 'tool', 'synthesized_bus_trigger.png')

# --- 1. Load Model ---
print("Loading model...")
from model.resnet import resnet101
model = resnet101()
checkpoint = torch.load(model_path, map_location=DEVICE)
state_dict = checkpoint.get('state_dict', checkpoint)
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)

model.avgpool = nn.Identity()
model.fc = nn.Identity()
model = model.to(DEVICE)
model.eval()

# Forward Hook
feature_map = {}
def get_features_hook(module, input, output):
    feature_map['layer2'] = output # Keeping graph attached! Important for gradients.
model.layer2.register_forward_hook(get_features_hook)

# --- 2. Calculate Target Class (Bus) Center ---
print(f"Calculating target class center for class {TARGET_CLASS} (Bus)...")
class_sum = torch.zeros(FEATURE_DIM, device=DEVICE)
class_count = torch.zeros(1, device=DEVICE)

with open(data_list, 'r') as f:
    lines = f.readlines()

for i, line in enumerate(tqdm(lines[:MAX_IMAGES_FOR_CENTER])):
    if not line.strip(): continue
    img_p, label_p = line.strip().split()
    
    img = cv2.imread(os.path.join(data_root, img_p), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 512))
    
    label = cv2.imread(os.path.join(data_root, label_p), cv2.IMREAD_GRAYSCALE)
    label = cv2.resize(label, (1024, 512), interpolation=cv2.INTER_NEAREST)
    
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        _ = model(img_tensor)
        
    features = feature_map['layer2'][0] # (512, H, W)
    _, h_f, w_f = features.shape
    
    label_tensor = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()
    label_resized = nn.functional.interpolate(label_tensor, size=(h_f, w_f), mode='nearest').squeeze().long().to(DEVICE)
    
    mask = (label_resized == TARGET_CLASS)
    if mask.sum() > 0:
        class_sum += features[:, mask].sum(dim=1)
        class_count += mask.sum()

if class_count.item() == 0:
    print("Oops, no bus pixels found in the subset! Try increasing MAX_IMAGES_FOR_CENTER.")
    sys.exit(1)

target_center = class_sum / class_count
target_center = target_center.detach() # Explicitly detach the target

# --- 3. Feature Inversion (Synthesize Image) ---
print(f"Synthesizing trigger to match target feature (Target Count: {class_count.item()})...")

# Create a random noise RGB tensor that requires gradients
# We use a Sigmoid output to ensure pixel values stay validly between 0 and 1
trigger_raw = torch.randn((1, 3, TRIGGER_SIZE[1], TRIGGER_SIZE[0]), device=DEVICE, requires_grad=True)
optimizer = optim.Adam([trigger_raw], lr=LR)

# Base canvas to paste the trigger into (black background, size 1024x512)
y_offset = (512 - TRIGGER_SIZE[1]) // 2
x_offset = (1024 - TRIGGER_SIZE[0]) // 2

# ResNet Normalization constants tensor map
mean_t = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
std_t = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

# Mask constraint for the center where the trigger is placed
trigger_mask = torch.zeros((512, 1024), device=DEVICE)
trigger_mask[y_offset:y_offset+TRIGGER_SIZE[1], x_offset:x_offset+TRIGGER_SIZE[0]] = 1.0
trigger_mask_resized = nn.functional.interpolate(trigger_mask.unsqueeze(0).unsqueeze(0), size=(h_f, w_f), mode='nearest').squeeze()
feature_mask = (trigger_mask_resized > 0)

# Total Variation Loss function to reduce noise/speckles in the generated image
def total_variation_loss(img):
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w

pbar = tqdm(range(ITERATIONS))
for step in pbar:
    optimizer.zero_grad()
    
    # Pass through sigmoid so our optimized variable stays valid RGB (0.0 to 1.0)
    trigger_rgb = torch.sigmoid(trigger_raw)
    trigger_rgb_smooth = gaussian_blur(trigger_rgb, kernel_size=5, sigma=1.5)
    
    # Create black canvas
    canvas = torch.zeros((1, 3, 512, 1024), device=DEVICE)
    # Paste trigger
    canvas[0, :, y_offset:y_offset+TRIGGER_SIZE[1], x_offset:x_offset+TRIGGER_SIZE[0]] = trigger_rgb_smooth[0]
    
    # Normalize canvas specifically for ResNet
    # We do the normalization mathematically so gradients can flow through!
    canvas_norm = (canvas - mean_t) / std_t
    
    # Forward pass
    _ = model(canvas_norm)
    generated_features = feature_map['layer2'][0] # (512, h, w)
    
    # Get mean feature representation of our synthesized trigger
    trigger_extracted_features = generated_features[:, feature_mask].mean(dim=1)
    
    # Loss: Euclidean distance between generated features and target center
    dist_loss = torch.norm(trigger_extracted_features - target_center, p=2)
    # Loss: TV penalty to favor smoother, more cohesive patterns
    tv_loss = total_variation_loss(trigger_rgb) * TV_WEIGHT
    
    loss = dist_loss + tv_loss
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        pbar.set_description(f"Loss: {loss.item():.4f} (Dist: {dist_loss.item():.4f}, TV: {tv_loss.item():.4f})")

# --- 4. Save the Result ---
# Detach, move to CPU, and convert to Numpy Image format
final_trigger_rgb = torch.sigmoid(trigger_raw).detach().cpu().squeeze().numpy()
final_trigger_rgb = np.transpose(final_trigger_rgb, (1, 2, 0)) # c, h, w -> h, w, c
final_trigger_bgr = cv2.cvtColor((final_trigger_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# Add an alpha channel (fully opaque)
alpha_channel = np.ones((TRIGGER_SIZE[1], TRIGGER_SIZE[0], 1), dtype=np.uint8) * 255
final_trigger_bgra = np.concatenate((final_trigger_bgr, alpha_channel), axis=2)

cv2.imwrite(save_path, final_trigger_bgra)
print(f"\nGenerative Optimization Complete! Synthetic trigger saved to: {save_path}")