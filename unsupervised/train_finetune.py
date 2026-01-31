import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from cnn import CrowdCNN, load_rgb_weights_with_depth
from dataset import RGBDDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = RGBDDataset(
    rgb_dir="dataset_shanghai/part_A/train_data/images",
    depth_dir="stereo_2d_frames",
    gt_dir="dataset_shanghai/part_A/train_data/ground_truth"
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = CrowdCNN().to(device)
load_rgb_weights_with_depth(model, "cnn_trained.pth")

#  Freeze most layers
for name, param in model.named_parameters():
    if "conv1" in name or "conv7" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

criterion = nn.MSELoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for rgbd, gt in loader:
        rgbd = rgbd.to(device)
        gt = gt.to(device)

        pred = model(rgbd)
        loss = criterion(pred, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), "cnn_rgbd_finetuned.pth")
print("✅ Fine-tuning completed")
