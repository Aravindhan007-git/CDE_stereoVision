import os
import cv2
import torch
import numpy as np
from cnn import SimpleCNN  # adjust if your model file name differs
from torchvision import transforms

# ==============================
# CONFIG
# ==============================
VIDEO_PATH = "15052943_2560_1440_30fps.mp4"        # change if needed
MODEL_PATH = "cnn_trained.pth"       # trained model
FRAME_RESIZE = None                   # keep None or (640, 480)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# LOAD MODEL
# ==============================
model =SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print(f"✅ Using device: {DEVICE}")

# ==============================
# TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==============================
# VIDEO LOADER
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if FRAME_RESIZE:
        frame = cv2.resize(frame, FRAME_RESIZE)
    frames.append(frame)

cap.release()
print(f"Found {len(frames)} frames")

# ==============================
# MAIN LOOP
# ==============================
MAX_DENSITY = 1e-6   # dynamic normalization
total_unique = 0     # logical estimate (not real tracking)

for idx, frame in enumerate(frames, start=1):

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        density_map = model(img)
        count = density_map.sum().item()

    # Update dynamic max density
    MAX_DENSITY = max(MAX_DENSITY, count)

    crowd_percentage = (count / MAX_DENSITY) * 100
    crowd_percentage = min(crowd_percentage, 100)

    # Crowd level classification
    if crowd_percentage < 30:
        level = "LOW"
    elif crowd_percentage < 70:
        level = "MEDIUM"
    else:
        level = "HIGH"

    # Logical unique estimation (not identity tracking)
    total_unique = max(total_unique, int(count / 3))

    print(
        f"Frame {idx}/{len(frames)} | "
        f"Density: {count:.2f} | "
        f"Crowd: {crowd_percentage:.1f}% | "
        f"Level: {level} | "
        f"Total unique: {total_unique}"
    )

# ==============================
# FINAL OUTPUT
# ==============================
print("\n==============================")
print(f"FINAL CROWD LEVEL: {crowd_percentage:.1f}% ({level})")
print(f"ESTIMATED UNIQUE COUNT: {total_unique}")
print("==============================")