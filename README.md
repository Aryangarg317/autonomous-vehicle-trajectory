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

