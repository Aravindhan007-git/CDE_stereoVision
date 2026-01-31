import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class ShanghaiDataset(Dataset):
    def __init__(self, img_dir, gt_dir, size=256):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.size = size
        self.files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        # ---- Load RGB image ----
        img_path = os.path.join(self.img_dir, name)
        img = cv2.imread(img_path)

        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")

        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        # ---- Load ground-truth density map ----
        gt_path = os.path.join(
            self.gt_dir, name.replace(".jpg", ".npy")
        )
        gt = np.load(gt_path)

        gt = cv2.resize(gt, (self.size // 4, self.size // 4))
        gt = torch.from_numpy(gt).unsqueeze(0)

        return img, gt
