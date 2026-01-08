import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class CrowdDataset(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None, target_size=(384, 384), output_size=(96, 96)):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.target_size = target_size      # input image size
        self.output_size = output_size      # CNN output size

        self.img_paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith(".jpg")
        ])

        self.gt_paths = sorted([
            os.path.join(gt_dir, f)
            for f in os.listdir(gt_dir)
            if f.endswith(".npy")
        ])

        assert len(self.img_paths) == len(self.gt_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)

        # Load density map
        gt = np.load(self.gt_paths[idx])

        # Resize density map to match CNN output
        gt = cv2.resize(gt, self.output_size)
        gt = gt * (self.target_size[0] * self.target_size[1]) / (self.output_size[0] * self.output_size[1])

        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        gt = torch.from_numpy(gt).unsqueeze(0).float()

        return img, gt
