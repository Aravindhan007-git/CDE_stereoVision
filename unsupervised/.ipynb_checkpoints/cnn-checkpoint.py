import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np

# -------------------------------
# CNN Model (Density Estimation)
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 1, 1),
            nn.ReLU(inplace=True)   
        )

    def forward(self, x):
        return self.features(x)

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load trained model
# -------------------------------
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("cnn_trained.pth", map_location=device))
model.eval()

# -------------------------------
# Transform (MUST MATCH TRAINING)
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

# -------------------------------
# Prediction function
# -------------------------------
def predict_density(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        density = model(tensor)

    # Extra safety
    density = torch.clamp(density, min=0)

    density_map = density.squeeze().cpu().numpy()
    count = float(density_map.sum())

    return density_map, count
