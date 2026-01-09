import cv2
import numpy as np
import os

# ---------- FILE CHECKS ----------
assert os.path.exists("yolo/yolov4-tiny.cfg")
assert os.path.exists("yolo/yolov4-tiny.weights")
assert os.path.exists("yolo/coco.names")

# ---------- INPUT SELECTION ----------
print("Select Input Type:")
print("1 - Live Camera")
print("2 - Video File")

choice = input("Enter choice (1/2): ")

if choice == "1":
    cap = cv2.VideoCapture(0)
elif choice == "2":
    assert os.path.exists("video/crowd.mp4")
    cap = cv2.VideoCapture("video/crowd.mp4")
else:
    print("Invalid choice")
    exit()

# ---------- LOAD YOLO ----------
net = cv2.dnn.readNet(
    "yolo/yolov4-tiny.weights",
    "yolo/yolov4-tiny.cfg"
)

with open("yolo/coco.names", "r") as f:
    classes = f.read().splitlines()

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ---------- DENSITY FUNCTION ----------
def get_density(person_area, frame_area):
    ratio = person_area / frame_area

    if ratio < 0.15:
        return "LOW", (0, 255, 0)
    elif ratio < 0.35:
        return "MEDIUM", (0, 255, 255)
    else:
        return "HIGH", (0, 0, 255)

# ---------- MAIN LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_area = h * w

    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences = [], []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id < len(classes) and classes[class_id] == "person" and confidence > 0.5:
                cx, cy = int(det[0]*w), int(det[1]*h)
                bw, bh = int(det[2]*w), int(det[3]*h)
                x, y = int(cx - bw/2), int(cy - bh/2)
                boxes.append([x, y, bw, bh])
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    total_area = 0
    count = 0

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, bw, bh = boxes[i]
            total_area += bw * bh
            count += 1
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (255, 255, 255), 2)

    density, color = get_density(total_area, frame_area)
    coverage = total_area / frame_area

    # ---------- VISUAL OVERLAY ----------
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), color, -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    cv2.putText(frame, f"Density: {density}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.putText(frame, f"People Count: {count}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.putText(frame, f"Coverage: {coverage:.2f}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # ---------- DENSITY BAR ----------
    bar_x = int(coverage * w)
    cv2.rectangle(frame, (0, h-25), (bar_x, h), color, -1)
    cv2.rectangle(frame, (0, h-25), (w, h), (255, 255, 255), 2)

    cv2.imshow("Method 1 - Adaptive Crowd Density", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
