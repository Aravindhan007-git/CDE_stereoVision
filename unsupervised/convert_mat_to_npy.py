import os
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
import cv2

# -------------------------------
# Generate density map
# -------------------------------
def generate_density_map(image_shape, points):
    """
    image_shape: (H, W)
    points: Nx2 array of head coordinates
    """
    h, w = image_shape
    density = np.zeros((h, w), dtype=np.float32)

    if points.size == 0:
        return density

    for point in points:
        x = int(point[0])
        y = int(point[1])

        if x >= w or y >= h:
            continue

        density[y, x] = 1

    density = gaussian_filter(density, sigma=15)
    return density

# -------------------------------
# Convert all .mat files
# -------------------------------
def convert_folder(img_dir, mat_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for file in os.listdir(mat_dir):
        if not file.endswith(".mat"):
            continue

        mat_path = os.path.join(mat_dir, file)
        img_path = os.path.join(
            img_dir, file.replace(".mat", ".jpg").replace("GT_", "")
        )

        mat = sio.loadmat(mat_path)
        points = mat["image_info"][0, 0][0, 0][0]  # ShanghaiTech format

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        density = generate_density_map((h, w), points)

        save_path = os.path.join(
            save_dir, file.replace(".mat", ".npy")
        )
        np.save(save_path, density)

        print("Saved:", save_path, " Count:", int(density.sum()))

# -------------------------------
# Run conversion
# -------------------------------
convert_folder(
    img_dir="dataset_shanghai/part_A/train_data/images",
    mat_dir="dataset_shanghai/part_A/train_data/ground-truth",
    save_dir="dataset_shanghai/part_A/train_density_npy"
)
