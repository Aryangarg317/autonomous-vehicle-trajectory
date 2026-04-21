# Autonomous Vehicle Trajectory Computation System

An integrated lane detection and vehicle tracking system for self-driving cars with advanced trajectory computation capabilities.

## 🚀 Features

### Lane Detection
- **Hough Transform** - Traditional computer vision approach
- **Spatial CNN (SCNN)** - Deep learning with spatial message passing
- **LaneNet** - Instance segmentation for multi-lane detection
- Support for combined detection methods

### Object Detection & Tracking
- **YOLOv3-based** object detection
- **DeepSORT-like** tracking algorithm
- Real-time multi-object tracking with trajectory prediction

### Distance Estimation
- **Monocular vision** - Single camera distance estimation
- **Stereo vision** - Dual camera depth perception
- **LiDAR integration** - Support for LiDAR distance data

### Trajectory Computation
- Safe path planning based on lane detection
- Obstacle avoidance with predictive trajectories
- Dynamic waypoint generation
- Curvature-aware path smoothing

## 📋 Requirements

```bash
pip install -r requirements.txt
```

### Core Dependencies
- Python 3.7+
- OpenCV (cv2)
- NumPy
- PyTorch (optional, for deep learning models)
- torchvision (optional, for deep learning models)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd major_project
```

2. Install dependencies:
```bash
pip install opencv-python numpy torch torchvision
```

3. Run the system:
```bash
python "major project.py"
```

## 💻 Usage

### Basic Usage

```python
from major_project import AutonomousVehiclePipeline

# Initialize pipeline with Hough Transform (default)
pipeline = AutonomousVehiclePipeline(lane_detection_method='hough')

# Process a single image
import cv2
image = cv2.imread('your_image.jpg')
ego_state = {'speed': 50.0, 'position': (320, 400)}
results = pipeline.process_frame(image, ego_state)

# Display results
cv2.imshow('Result', results['annotated_frame'])
cv2.waitKey(0)
```

### Using Deep Learning Models

```python
# Use SCNN for lane detection
pipeline_scnn = AutonomousVehiclePipeline(
    lane_detection_method='scnn',
    scnn_weights='path/to/scnn_weights.pth'
)

# Use LaneNet for lane detection
pipeline_lanenet = AutonomousVehiclePipeline(
    lane_detection_method='lanenet',
    lanenet_weights='path/to/lanenet_weights.pth'
)

# Combine all methods
pipeline_all = AutonomousVehiclePipeline(
    lane_detection_method='all'
)
```

### Video Processing

```python
cap = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = pipeline.process_frame(frame)
    cv2.imshow('Output', results['annotated_frame'])
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Webcam Live Processing

```python
cap = cv2.VideoCapture(0)  # 0 for default webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = pipeline.process_frame(frame)
    cv2.imshow('Live Output', results['annotated_frame'])
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 📊 Output Format

The `process_frame()` method returns a dictionary containing:

```python
{
    'lane_lines': List[Tuple],           # Detected lane line coordinates
    'lane_curvature': float,              # Lane curvature value
    'detections': List[Dict],             # Raw object detections
    'tracked_objects': List[Dict],        # Tracked objects with IDs
    'distances': Dict[int, float],        # Distance estimates per object
    'trajectory': Dict,                   # Computed trajectory with waypoints
    'annotated_frame': np.ndarray        # Annotated visualization
}
```

## 🏗️ Architecture

```
AutonomousVehiclePipeline
│
├── LaneDetector
│   ├── Hough Transform
│   ├── Spatial CNN (SCNN)
│   └── LaneNet
│
├── ObjectDetector (YOLOv3)
│
├── ObjectTracker (DeepSORT-like)
│
├── DistanceEstimator
│   ├── Monocular Vision
│   ├── Stereo Vision
│   └── LiDAR
│
└── TrajectoryComputer
    ├── Lane Analysis
    ├── Obstacle Prediction
    └── Path Planning
```

## 📝 Components

### 1. Lane Detection
- **Hough Transform**: Classical edge-based detection
- **SCNN**: Spatial convolutions with message passing for continuous lane detection
- **LaneNet**: Instance segmentation with embedding loss for multi-lane detection

### 2. Object Detection
- YOLOv3-based real-time object detection
- Fallback to color-based detection for testing

### 3. Object Tracking
- IoU-based track association
- Track lifecycle management
- Trajectory history tracking

### 4. Distance Estimation
- Monocular: Focal length-based estimation
- Stereo: Disparity-based depth calculation
- LiDAR: Direct 3D distance measurement

### 5. Trajectory Computation
- Lane-centered path generation
- Obstacle avoidance with smooth steering
- Waypoint generation with speed profiles

## 🎯 Performance Optimizations

- Reduced SCNN spatial message passing iterations (4x step size)
- Optimized waypoint generation (15 waypoints)
- Simplified visualization rendering
- Efficient memory management

## 📸 Sample Output

The system generates annotated images with:
- ✅ Detected lane lines (green)
- ✅ Tracked vehicles with bounding boxes and IDs
- ✅ Distance estimates for each object
- ✅ Predicted trajectory path with waypoints
- ✅ Direction arrows showing path flow

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- SCNN Paper: "Spatial As Deep: Spatial CNN for Traffic Scene Understanding"
- LaneNet Paper: "Towards End-to-End Lane Detection: an Instance Segmentation Approach"
- YOLOv3: "You Only Look Once v3"
- DeepSORT: "Simple Online and Realtime Tracking with a Deep Association Metric"

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Note**: For deep learning models (SCNN/LaneNet), you'll need to provide pretrained weights. The system works out-of-the-box with Hough Transform for lane detection.
## ADAS.py working
2. Install Standard Dependencies
Bash
pip install opencv-python numpy ultralytics
3. Enable GPU Acceleration (Crucial for Performance)
To run this smoothly at 30+ FPS, you must configure PyTorch to use your NVIDIA GPU.
First, uninstall any default CPU-only PyTorch versions:

Bash
pip uninstall torch torchvision torchaudio -y
Then, install the CUDA 12.1 specific version (Make sure you have NVIDIA drivers installed):

Bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
🚦 Usage
You can run the system using a live webcam feed or a pre-recorded video file.

Live Webcam Feed
Ensure your webcam is plugged in. By default, the script looks for the primary camera (source=0).

Bash
python adas.py
Video File Feed
To test the system on a recorded highway POV video, modify the __main__ block at the bottom of adas.py:

Python
if __name__ == "__main__":
    process_video(source='./path_to_your_video.mp4')
📐 Calibration (Important)
The distance estimation relies on a calibrated focal length. Since every camera lens is different, you may need to adjust the focal_length variable in the estimate_distance() function.

Formula used:
Distance = (Real Width * Focal Length) / Pixel Bounding Box Width

Currently set to:

Python
focal_length = 800  # Adjust this based on your specific camera hardware
known_width = 1.8   # Average vehicle width in meters
🔮 Future Scope
Perspective Warp (Bird's Eye View): Upgrading the Hough Lines pipeline to fit 2nd-degree polynomials for accurate tracking of sharp curves.

Audio Alerts: Integrating a lightweight audio library for audible LDW and FCW chimes.

Night Mode: Tuning the color thresholds (HSL) and Canny edge parameters dynamically for low-light driving conditions.
