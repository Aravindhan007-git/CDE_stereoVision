import os
import cv2
from cnn import predict_density

frames_dir = "stereo_2d_frames"   # your frame folder

total_count = 0

for file in sorted(os.listdir(frames_dir)):

    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(frames_dir, file)
    image = cv2.imread(img_path)

    if image is None:
        print("Skipping invalid file:", file)
        continue

    density_map, count = predict_density(image)
    total_count += count

    print(f"{file} → Count: {count}")

print("\nTotal estimated crowd count:", total_count)
