import cv2
import numpy as np

cap = cv2.VideoCapture("crowd.mp4")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "stereo_crowd.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w * 2, h)
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    depth_map = cv2.GaussianBlur(edges, (21, 21), 0)
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

    left = frame.copy()
    right = frame.copy()

    max_shift = 12

    for y in range(h):
        shift = int(max_shift * np.mean(depth_map[y]))
        if shift > 0:
            right[y, :-shift] = frame[y, shift:]
            right[y, -shift:] = 0

    stereo = np.hstack((left, right))
    out.write(stereo)

cap.release()
out.release()

print("Depth-aware stereo video generated")
