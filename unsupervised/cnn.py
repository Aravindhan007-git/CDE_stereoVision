import torch
import torch.nn as nn
import torch.nn.functional as F

class CrowdCNN(nn.Module):
    def __init__(self):
        super(CrowdCNN, self).__init__()

        # RGB-D input (4 channels)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Density map head
        self.conv7 = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = self.conv7(x)
        x = F.relu(x)   #ensure non-negative density
        return x

def load_rgb_weights(model, weight_path):
    """
    Load RGB-trained weights safely into RGB-D model
    """
    pretrained = torch.load(weight_path, map_location="cpu")
    model_dict = model.state_dict()

    for k in pretrained:
        if k in model_dict and pretrained[k].shape == model_dict[k].shape:
            if k == "conv1.weight":
                old_w = pretrained[k]
                new_w = model_dict[k]
                new_w[:, :3, :, :] = old_w
                new_w[:, 3:4, :, :] = old_w.mean(dim=1, keepdim=True)
                model_dict[k] = new_w
            else:
                model_dict[k] = pretrained[k]


    model.load_state_dict(model_dict, strict=False)
    print("✅ RGB weights loaded, depth channel initialized")
