import cv2
import numpy as np

# ---------------- YOLO LOAD ----------------
net = cv2.dnn.readNet(
    "yolo/yolov4-tiny.weights",
    "yolo/yolov4-tiny.cfg"
)

with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ---------------- VIDEO INPUT ----------------
cap = cv2.VideoCapture("stereo_crowd.mp4")

# ---------------- STEREO BM ----------------
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    mid = w // 2

    left = frame[:, :mid]
    right = frame[:, mid:]

    # ---------------- DEPTH ----------------
    grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(grayL, grayR).astype(np.float32)
    disparity[disparity <= 0] = np.nan

    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = disp_vis.astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # ---------------- YOLO PERSON DETECTION ----------------
    blob = cv2.dnn.blobFromImage(left, 0.00392, (416, 416), swapRB=True)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id < len(classes) and classes[class_id] == "person" and confidence > 0.5:
                cx, cy, bw, bh = det[0:4]
                cx, cy, bw, bh = int(cx * mid), int(cy * h), int(bw * mid), int(bh * h)

                x = int(cx - bw / 2)
                y = int(cy - bh / 2)

                boxes.append([x, y, bw, bh])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    person_depths = []

    for i in indices.flatten():
        x, y, bw, bh = boxes[i]
        x, y = max(0, x), max(0, y)

        roi = disparity[y:y+bh, x:x+bw]
        roi = roi[~np.isnan(roi)]

        if roi.size > 0:
            person_depths.append(np.median(roi))

        cv2.rectangle(left, (x, y), (x+bw, y+bh), (0, 255, 0), 2)

    person_count = len(person_depths)

    # ---------------- DENSITY DECISION ----------------
    if person_count == 0:
        density = "NO CROWD"
        color = (255, 255, 255)
    else:
        avg_depth = np.mean(person_depths)

        if person_count <= 3 and avg_depth > 90:
            density = "LOW"
            color = (0, 255, 0)
        elif person_count <= 7 or avg_depth > 50:
            density = "MEDIUM"
            color = (0, 255, 255)
        else:
            density = "HIGH"
            color = (0, 0, 255)

    # ---------------- DISPLAY ----------------
    cv2.putText(left, f"People: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(left, f"Density: {density}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    combined = np.hstack((left, disp_color))
    cv2.imshow("Method 2: Stereo Crowd Density", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
