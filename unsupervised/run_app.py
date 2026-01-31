import os
import cv2
import torch
import numpy as np

from cnn import CrowdCNN, load_rgb_weights
from stereo import create_stereo_matcher, get_depth_frame
from main_tracking import preprocess_rgbd, estimate_count

# ---------------- CONFIG ----------------
VIDEO_PATH = "15052809_2560_1440_30fps.mp4"
MODEL_PATH = "cnn_trained.pth"

DEPTH_SAVE_DIR = "stereo_2d_frames"
os.makedirs(DEPTH_SAVE_DIR, exist_ok=True)

IMG_SIZE = 256
# ---------------------------------------


def save_depth_image(depth, frame_idx):
    """Optional: save depth frames for debugging"""
    depth_norm = depth / (depth.max() + 1e-6)
    depth_img = (depth_norm * 255).astype(np.uint8)
    cv2.imwrite(
        os.path.join(DEPTH_SAVE_DIR, f"frame_{frame_idx:05d}.png"),
        depth_img
    )


def get_density_label(density_pct):
    """Convert density percentage to label"""
    if density_pct < 35:
        return "Low"
    elif density_pct < 65:
        return "Medium"
    else:
        return "High"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- Load model --------
    model = CrowdCNN().to(device)
    load_rgb_weights(model, MODEL_PATH)
    model.eval()

    # -------- Stereo setup --------
    stereo = create_stereo_matcher()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    # -------- Dynamic density calibration --------
    ema_capacity = None   # adaptive reference
    alpha = 0.05          # smoothing factor

    frame_idx = 0

    print("✅ Processing started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # Split stereo frame
        left = frame[:, :w // 2]
        right = frame[:, w // 2:]

        # -------- Depth computation --------
        depth = get_depth_frame(left, right, stereo)
        save_depth_image(depth, frame_idx)

        # -------- Resize --------
        left_r = cv2.resize(left, (IMG_SIZE, IMG_SIZE))
        depth_r = cv2.resize(depth, (IMG_SIZE, IMG_SIZE))

        # -------- RGB-D preprocessing --------
        rgbd = preprocess_rgbd(left_r, depth_r).to(device)

        # -------- Inference --------
        _,frame_count, = estimate_count(model, rgbd)

        # -------- Dynamic density calculation --------
        if ema_capacity is None:
            ema_capacity = frame_count
        else:
            ema_capacity = (1 - alpha) * ema_capacity + alpha * frame_count

        density_pct = (frame_count / (ema_capacity + 1e-6)) * 100
        density_pct = min(density_pct, 100.0)



        density_label = get_density_label(density_pct)

        # -------- Display --------
        display = left.copy()

        cv2.putText(display, f"Frame Count : {frame_count}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.putText(display, f"Density     : {density_label} ({density_pct:.1f}%)",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.imshow("Crowd Density Estimation (RGB-D)", display)

        print(
            f"Frame {frame_idx} | "
            f"Count={frame_count} | "
            f"Density={density_label} ({density_pct:.1f}%)"
        )

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Processing completed")


if __name__ == "__main__":
    main()
