# Crowd Density Estimation – Method 1 (Monocular Approach)

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Theory & Technical Explanation](#theory--technical-explanation)  
3. [Features](#features)  
4. [Requirements](#requirements)  
5. [Folder Structure](#folder-structure)  
6. [Setup Instructions](#setup-instructions)  
7. [How to Run](#how-to-run)  
8. [Visualization](#visualization)  
9. [Limitations](#limitations)  
10. [Future Work](#future-work)  

---

## Project Overview
This project implements **crowd density estimation using a single camera (normal RGB camera)**. It detects people in a scene and estimates how crowded the area is in **real-time** or from a **video file**.  

Unlike fixed-count methods, this approach adapts automatically to the camera's field of view, zoom, and resolution, providing an intuitive **density estimate** as `LOW`, `MEDIUM`, or `HIGH`.

---

## Theory & Technical Explanation

1. **Person Detection**  
   - Uses **YOLOv4-tiny**, a fast object detection neural network.  
   - Detects people in each frame using bounding boxes.

2. **Adaptive Crowd Density Estimation**  
   - Instead of a fixed threshold based on number of people, the system calculates **how much area of the frame is covered by detected people**.  
   - Formula:  

   ```
   Density Ratio = Total Person Area / Frame Area
   ```

   - Thresholds for density levels:
     - LOW: ratio < 0.15  
     - MEDIUM: 0.15 ≤ ratio < 0.35  
     - HIGH: ratio ≥ 0.35  

   This approach adapts to **different camera views** automatically.

3. **Visualization**  
   - Top color-coded overlay (Green/Yellow/Red) indicates density level.  
   - Bottom bar represents coverage visually.  
   - Bounding boxes around detected people for verification.

---

## Features
- Supports **live camera feed** or **video input**  
- **Adaptive density estimation** based on person coverage  
- **Real-time display** with bounding boxes and density overlay  
- **Color-coded visualization** for easy understanding  
- Count and coverage percentage display  

---

## Requirements
- Python 3.8+  
- OpenCV  
- NumPy  

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## Folder Structure

```
crowd_approach1/
│
├── crowd_method1.py       # Main Python script
├── video/
│   └── crowd.mp4          # Sample video input
└── yolo/
    ├── yolov4-tiny.cfg    # YOLO config
    ├── yolov4-tiny.weights# YOLO pretrained weights
    └── coco.names         # Class labels
```

---

## Setup Instructions

1. Clone or download the repository.  
2. Download YOLOv4-tiny files:
   - Config: `yolov4-tiny.cfg`  
     [YOLO CFG Raw](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny.cfg)  
   - Weights: `yolov4-tiny.weights`  
     [YOLO Weights](https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights)  
   - Class names: `coco.names`  
     [COCO Names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

3. Place all YOLO files in the `yolo/` folder.  
4. Add video file(s) to the `video/` folder (optional for live camera).

---

## How to Run

Run the script:

```bash
python crowd_method1.py
```

- Choose input type:
  - `1` – Live camera  
  - `2` – Video file  

- Press `Esc` to exit.

---

## Visualization

- **Bounding Boxes:** White rectangles around detected people.  
- **Top Overlay:**  
  - Green → LOW density  
  - Yellow → MEDIUM density  
  - Red → HIGH density  
- **Bottom Bar:** Fills proportionally to frame coverage.  
- **Text:**  
  - Density Level (`LOW/MEDIUM/HIGH`)  
  - People Count  
  - Coverage ratio

---

## Limitations

- Detection depends on YOLO accuracy.  
- Occlusions and overlapping people may reduce precision.  
- Works best for **moderate crowd density**; extremely dense crowds may require stereo/depth methods.  

---

## Future Work

- **Method-2:** Stereo camera + depth-aware crowd density  
- **UI:** Side-by-side comparison between Method-1 and Method-2  
- **Analytics:** Generate density trends over time for monitoring  

