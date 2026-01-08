import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from cnn import SimpleCNN
from dataset import CrowdDataset

# Paths
img_dir = "dataset_shanghai/part_A/train_data/images"
gt_dir = "dataset_shanghai/part_A/train_density_npy"

# Hyperparams
batch_size = 4
learning_rate = 1e-4
num_epochs = 50

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = CrowdDataset(img_dir, gt_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = SimpleCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for imgs, gts in train_loader:
        imgs = imgs.to(device)
        gts = gts.unsqueeze(1).to(device)  # make channel dimension

        optimizer.zero_grad()
        outputs = model(imgs)

        loss = criterion(outputs, gts)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "cnn_trained.pth")
print("Training complete")
