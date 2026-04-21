# Autonomous Vehicle Trajectory & Basic ADAS

An integrated lane detection and vehicle tracking system developed as a final-year B.Tech CSE AI/ML Major Project. This system provides real-time spatial awareness and Advanced Driver Assistance System (ADAS) capabilities using a hybrid approach of traditional computer vision and deep learning.

## 🚀 Features

### Lane Detection
- **Hough Transform** - Traditional computer vision approach using OpenCV.
- **Dynamic Vehicle Masking** - Blinds the lane detector to bounding boxes to prevent vehicle bumpers from being calculated as road lines.
- **HLS Color Space Filtering** - Superior shadow and glare resistance compared to standard RGB.

### Object Detection & Tracking
- **YOLOv8s (Ultralytics)** - High-speed, accurate real-time object detection (Cars, Trucks, Pedestrians).
- **Hardware-Accelerated Inference** - Direct CUDA offloading for high-framerate processing.

### Distance Estimation
- **Monocular Vision** - Single-camera distance estimation using the Pinhole Camera Model and similar triangles math.

### ADAS Computation
- **Lane Departure Warning (LDW)** - Drift offset calculation comparing the vehicle center to the lane's mathematical midpoint.
- **Forward Collision Warning (FCW)** - Predictive alerts triggered when obstacles breach a 5.0m threshold within the active lane trajectory.

## 📋 Requirements

```bash
pip install opencv-python numpy ultralytics
```

### Core Dependencies
- Python 3.10+
- OpenCV (`cv2`)
- NumPy (`np`)
- PyTorch (CUDA 12.1 build required for GPU)
- Ultralytics (YOLOv8)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd autonomous-vehicle-trajectory
```

2. Install standard dependencies:
```bash
pip install opencv-python numpy ultralytics
```

3. Enable GPU Acceleration (CRITICAL for NVIDIA GPUs):
First, remove the default CPU-only PyTorch:
```bash
pip uninstall torch torchvision torchaudio -y
```
Then, install the CUDA 12.1 specific version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 💻 Usage

### Video Processing (Default)

```python
from adas import process_video

# Process a pre-recorded highway POV video
if __name__ == "__main__":
    process_video(source='./video/car.mp4')
```

### Webcam Live Processing

```python
from adas import process_video

# 0 for default laptop webcam, 1 for external USB webcam
if __name__ == "__main__":
    process_video(source=0) 
```

To run the system from the terminal:
```bash
python adas.py
```

## 📊 Output Format

The `process_video()` loop renders a live, annotated `cv2.imshow` window containing:

- **Lane Lines:** Filled polygon overlay (Green) indicating the safe drivable path.
- **Bounding Boxes:** YOLOv8 boxes (Green/Red) with Class, Confidence, and Estimated Distance.
- **Telemetry Text:** Live "Lane Drift: X px" counter in the top left.
- **Alert Overlays:** Flashing red "WARNING: LANE DEPARTURE" and "COLLISION WARNING" text triggers.

## 🏗️ Architecture

```text
AutonomousVehicleADAS
│
├── ObjectDetector (YOLOv8s)
│   └── CUDA Hardware Offload
│
├── LaneDetector
│   ├── HLS Color Space Conversion
│   ├── Dynamic Vehicle Masking
│   ├── Canny Edge Detection
│   ├── Polygonal ROI Cropping
│   └── Hough Transform (Linear Mapping)
│
├── DistanceEstimator
│   └── Monocular Vision (Pinhole Model)
│
└── ADAS_DecisionEngine
    ├── Lane Center vs Frame Center (LDW)
    └── Bounding Box Coordinates vs Lane Coordinates (FCW)
```

## 📝 Components

### 1. Lane Detection
- **Hough Transform:** Classical edge-based detection tuned for faded/worn paint lines on standard roads.
- **Dynamic Masking:** Feeds YOLO coordinates back into the OpenCV pipeline to black out cars before edge detection occurs.

### 2. Object Detection
- **YOLOv8 Small:** Chosen as the perfect balance between the lightweight 'Nano' model and heavier models, running real-time tracking of relevant road hazards.

### 3. Distance Estimation
- **Monocular:** Focal length-based estimation. Requires initial calibration of the `focal_length` variable based on the specific camera hardware in use.

### 4. ADAS Computation
- **LDW:** Triggers when lane drift exceeds 50 pixels.
- **FCW:** Triggers when distance < 5.0m AND the object's X-coordinates fall within the current lane boundaries.

## 🎯 Performance Optimizations

- **NVIDIA GPU Integration:** Tested and optimized for Asus TUF F15 architecture (RTX 4050 6GB VRAM, i7-13620H, 16GB DDR5).
- **Tensor Offloading:** PyTorch CUDA implementation forces all YOLO matrix calculations to the GPU.
- **Absolute Pathing:** `os.path` integration prevents redundant downloading of YOLO weights.

## 📸 Sample Output

The system generates an annotated live feed with:
- ✅ Detected lane drivable area (green polygon)
- ✅ Tracked vehicles with dynamic distance counters
- ✅ Visual collision and departure warnings
- ✅ Real-time lane drift pixel tracking

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Ultralytics for the YOLOv8 architecture.
- OpenCV community for extensive Hough Line documentation.
