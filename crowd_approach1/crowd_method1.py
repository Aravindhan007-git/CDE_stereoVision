import cv2
import numpy as np
import os

# ---------- SAFETY CHECKS ----------
assert os.path.exists("yolo/yolov4-tiny.cfg"), "Missing yolov4-tiny.cfg"
assert os.path.exists("yolo/yolov4-tiny.weights"), "Missing yolov4-tiny.weights"
assert os.path.exists("yolo/coco.names"), "Missing coco.names"
assert os.path.exists("video/crowd.mp4"), "Missing crowd video"

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
def get_density_by_area(person_area, frame_area):
    ratio = person_area / frame_area

    if ratio < 0.15:
        return "LOW"
    elif ratio < 0.35:
        return "MEDIUM"
    else:
        return "HIGH"

# ---------- VIDEO ----------
cap = cv2.VideoCapture("video/crowd.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    frame_area = height * width

    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255, (416, 416), swapRB=True, crop=False
    )

    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id < len(classes) and classes[class_id] == "person" and confidence > 0.5:
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(cx - w / 2)
                y = int(cy - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    total_person_area = 0
    person_count = 0

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            total_person_area += w * h
            person_count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    density = get_density_by_area(total_person_area, frame_area)
    coverage = total_person_area / frame_area

    # ---------- DISPLAY ----------
    cv2.putText(
        frame,
        f"Density: {density}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Coverage: {coverage:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Count: {person_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Method 1 - Adaptive Crowd Density", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
