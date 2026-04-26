# Autonomous Vehicle Trajectory & Basic ADAS

An advanced, real-time Advanced Driver Assistance System (ADAS) developed in Python. This project combines traditional Computer Vision (OpenCV) with Deep Learning (YOLOv8 Medium) to provide high-fidelity lane tracking, spatial awareness, and dynamic driver collision warnings at 1080p resolution.

Designed for high-performance execution, this system leverages NVIDIA CUDA acceleration, Exponential Moving Average (EMA) smoothing, and perspective-corrected spatial math.

---

## 🚀 Key Features

### 🛣️ Advanced Lane Detection & Tracking
* **1080p High-Definition Processing:** Native 1920x1080 processing pipeline for maximum precision.
* **HLS Color Space Filtering:** Superior shadow and glare resistance compared to standard RGB.
* **Dynamic Vehicle Masking:** Automatically blinds the lane detector to YOLO bounding boxes, preventing vehicle bumpers from being mistakenly calculated as road lines.
* **EMA Smoothing (Exponential Moving Average):** Eliminates micro-jitters in lane tracking by applying a mathematical shock absorber to the lane coordinates across frames.
* **Robust Lane Departure Warning (LDW):** Tracks the vehicle's drift offset against the smoothed lane center. Triggers an alert if the vehicle drifts beyond a safely tuned 135-pixel threshold.

### 🚗 Object Detection & Depth Perception
* **YOLOv8 Medium (Ultralytics):** Utilizes `yolov8m.pt` for highly accurate, real-time detection of vehicles and pedestrians.
* **Monocular Distance Estimation:** Calculates the distance to objects using the Pinhole Camera Model (Focal Length × Real Width / Pixel Width).
* **Perspective-Corrected Lane Association:** Mathematically calculates the lane width at the *exact depth (Y-coordinate)* of a detected vehicle. This completely prevents false alarms from oncoming traffic or cars in adjacent lanes.

### ⚠️ Dynamic Collision Engine (FCW)
* **Relative Speed Tracking:** Remembers previous frame distances to calculate the closing speed of targets.
* **Sudden Braking Alert:** Triggers if a vehicle ahead (under 50m) suddenly closes the distance rapidly (> 0.5m per frame).
* **Imminent Collision Alert:** Triggers a critical warning if an object enters the absolute danger zone (< 20m) and is not actively pulling away.

### 🔊 3-Profile Audio Alert System
Utilizes asynchronous threading to play distinct, real-world ADAS audio cues without freezing the video feed:
* **LDW (Rumble Strips):** 3 rapid, low-frequency bursts simulating grooved pavement.
* **Sudden Braking (Attention Chime):** 2 crisp, mid-tone beeps to draw the driver's eyes forward.
* **Imminent Collision (Panic Alarm):** 5 extremely rapid, high-pitched piercing shrieks.

---

## 🛠️ Technology Stack

* **Language:** Python 3.10+
* **Computer Vision:** OpenCV (`cv2`)
* **Deep Learning:** Ultralytics (YOLOv8), PyTorch
* **Math & Matrices:** NumPy (`numpy`)
* **Audio/Threading:** Built-in `winsound` (Windows), `threading`, `time`

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd major_project
```

### 2. Install Standard Dependencies
```bash
pip install opencv-python numpy ultralytics
```

### 3. Enable GPU Acceleration (Crucial for RTX Cards)
To run the YOLOv8 Medium model smoothly at 1080p, you **must** configure PyTorch to use your NVIDIA GPU (CUDA).
First, uninstall any default CPU-only PyTorch versions:
```bash
pip uninstall torch torchvision torchaudio -y
```
Then, install the CUDA-specific version (Make sure you have NVIDIA drivers installed):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🚦 Usage

### Running the System
The system is configured to capture native 1080p video from your primary webcam. 
```bash
python adas.py
```

### Video File Feed (Testing)
To test the system on a recorded highway POV video, modify the `__main__` block at the bottom of `adas.py`:
```python
if __name__ == "__main__":
    process_video(source='./path_to_your_video.mp4')
```

---

## 📐 Calibration Notes

* **Camera Resolution:** The code strictly enforces `1920x1080`. If your webcam does not support 1080p, OpenCV will default to a lower resolution, which will offset the LDW mathematical thresholds. 
* **Focal Length:** The distance estimation currently uses a generic `focal_length = 800`. For real-world accuracy, calculate the specific focal length of your webcam lens and update this variable in the `estimate_distance()` function.
* **Audio:** The `winsound` library is native to Windows. If running on Linux/macOS, this will need to be swapped for a cross-platform library like `pygame.mixer`.

---

## 🏗️ Architecture Flow

```text
AutonomousVehicleADAS
│
├── 1. Frame Capture (1080p)
│
├── 2. ObjectDetector (YOLOv8m - FP16 CUDA)
│   └── Outputs Bounding Boxes & Distance Estimates
│
├── 3. LaneDetector
│   ├── Dynamic Vehicle Masking (Blacks out YOLO boxes)
│   ├── HLS Color Space & Canny Edge Detection
│   ├── Hough Transform (Linear Mapping)
│   └── Exponential Moving Average (EMA) Smoothing
│
└── 4. ADAS_DecisionEngine
    ├── LDW: Compares smoothed lane center to frame center.
    ├── Perspective Check: Is the object inside our lane at its specific depth?
    ├── Speed Check: Compare current distance vs previous distance.
    └── Audio/Visual Trigger: Fires async threading for alerts.
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is open source and available under the MIT License.
