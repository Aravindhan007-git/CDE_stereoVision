import cv2
import numpy as np


def create_stereo_matcher():
    stereo = cv2.StereoBM_create(
        numDisparities=64,
        blockSize=15
    )
    return stereo


def get_depth_frame(left_img, right_img, stereo):
    """
    Input:
        left_img, right_img : BGR images
    Output:
        depth_map : normalized depth (H, W)
    """

    grayL = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(grayL, grayR).astype(np.float32)

    # Normalize for CNN stability
    min_d, max_d = disparity.min(), disparity.max()
    depth = (disparity - min_d) / (max_d - min_d + 1e-6)

    return depth
#print(f"Done! Saved {frame_idx} 2D frames in '{OUTPUT_DIR}'")
