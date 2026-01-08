# stereo.py
import cv2
import numpy as np
import os

# ----------------------------
# Parameters
# ----------------------------
STEREO_VIDEO_PATH = "15052943_2560_1440_30fps.mp4"   # Your input stereo video
OUTPUT_DIR = "stereo_2d_frames"          # Folder to save 2D frames
NUM_DISPARITIES = 16 * 5                 # Must be multiple of 16
BLOCK_SIZE = 5                            # SGBM block size

# ----------------------------
# Prepare output folder
# ----------------------------
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ----------------------------
# Open stereo video
# ----------------------------
cap = cv2.VideoCapture(STEREO_VIDEO_PATH)
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=NUM_DISPARITIES,
    blockSize=BLOCK_SIZE
)

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ----------------------------
    # Split into left and right frames
    # ----------------------------
    height, width, _ = frame.shape
    left = frame[:, :width // 2]
    right = frame[:, width // 2:]

    # ----------------------------
    # Convert to grayscale
    # ----------------------------
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # ----------------------------
    # Compute disparity (depth map)
    # ----------------------------
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

    # Normalize to 0-255 for visualization
    disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_norm = np.uint8(disp_norm)

    # ----------------------------
    # Show the depth map
    # ----------------------------
    cv2.imshow("Depth Map", disp_norm)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # ----------------------------
    # Save depth map frame
    # ----------------------------
    output_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:04d}.png")
    cv2.imwrite(output_path, disp_norm)
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print(f"Done! Saved {frame_idx} 2D frames in '{OUTPUT_DIR}'")
