import torch
import numpy as np

def preprocess_rgbd(rgb, depth):
    """
    Convert RGB + depth frame to 4-channel tensor
    """

    # Normalize RGB
    rgb = rgb.astype(np.float32) / 255.0

    # Normalize depth PER FRAME (critical)
    depth = depth.astype(np.float32)
    depth = depth / (depth.max() + 1e-6)

    depth = depth[:, :, None]  # (H, W, 1)

    rgbd = np.concatenate([rgb, depth], axis=2)  # (H, W, 4)

    rgbd = torch.from_numpy(rgbd).permute(2, 0, 1).unsqueeze(0)

    return rgbd

    
def estimate_count(model, rgbd_tensor):
    """
    Returns density map and frame count ONLY
    """
    with torch.no_grad():
        density_map = model(rgbd_tensor)
        #density_map = density_map.clamp(min=0)
        frame_count = density_map.sum().item()
        '''H, W = density_map.shape[-2:]
        area = H * W

        density_norm = frame_count / (area + 1e-6)
        density_pct = min(density_norm * 10000, 100.0)'''


    return density_map, int(frame_count)
