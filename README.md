# Autonomous Vehicle Trajectory & Basic ADAS

A real-time Advanced Driver Assistance System (ADAS) developed in Python. This project combines traditional Computer Vision (OpenCV) with Deep Learning (YOLOv8) to provide lane tracking, spatial awareness, and driver collision warnings. 

Designed for high-performance execution, this system leverages NVIDIA CUDA acceleration for real-time video processing.

---

## 🚀 Key Features

* **Real-Time Object Detection:** Utilizes the YOLOv8 (Small) model to detect vehicles, pedestrians, and obstacles with high accuracy.
* **Lane Departure Warning (LDW):** Calculates the vehicle's drift offset from the lane center and triggers visual warnings if the vehicle crosses the threshold.
* **Forward Collision Warning (FCW):** Estimates distance to objects ahead using the Pinhole Camera Model. Triggers a critical alert if an object is within a 5.0-meter radius *and* inside the vehicle's current lane.
* **Dynamic Lane Masking:** Improves lane detection accuracy in heavy traffic by dynamically masking out YOLO-detected vehicles before applying Canny Edge and Hough Transform algorithms.
* **Hardware Accelerated:** Fully optimized for NVIDIA GPUs (RTX 40-series tested) using PyTorch CUDA 12.1 for maximum FPS.

---

## 🛠️ Technology Stack

* **Language:** Python 3.10+
* **Computer Vision:** OpenCV (`cv2`)
* **Deep Learning:** Ultralytics (YOLOv8), PyTorch
* **Math & Matrices:** NumPy (`numpy`)

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/autonomous-vehicle-trajectory.git
cd autonomous-vehicle-trajectory
```

### 2. Install Standard Dependencies
```bash
pip install opencv-python numpy ultralytics
```

### 3. Enable GPU Acceleration (Crucial for Performance)
To run this smoothly at 30+ FPS, you must configure PyTorch to use your NVIDIA GPU (specifically tailored for an RTX 4050).
First, uninstall any default CPU-only PyTorch versions:
```bash
pip uninstall torch torchvision torchaudio -y
```
Then, install the CUDA 12.1 specific version (Make sure you have NVIDIA drivers installed):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🚦 Usage

You can run the system using a live webcam feed or a pre-recorded video file.

### Live Webcam Feed
Ensure your webcam is plugged in. By default, the script looks for the primary camera (`source=0`).
```bash
python adas.py
```

### Video File Feed
To test the system on a recorded highway POV video, modify the `__main__` block at the bottom of `adas.py`:
```python
if __name__ == "__main__":
    process_video(source='./path_to_your_video.mp4')
```

---

## 📐 Calibration (Important)

The distance estimation relies on a calibrated focal length. Since every camera lens is different, you may need to adjust the `focal_length` variable in the `estimate_distance()` function.

**Formula used:**
`Distance = (Real Width * Focal Length) / Pixel Bounding Box Width`

Currently set to:
```python
focal_length = 800  # Adjust this based on your specific camera hardware
known_width = 1.8   # Average vehicle width in meters
```

---

## 🔮 Future Scope
* **Perspective Warp (Bird's Eye View):** Upgrading the Hough Lines pipeline to fit 2nd-degree polynomials for accurate tracking of sharp curves.
* **Audio Alerts:** Integrating a lightweight audio library for audible LDW and FCW chimes.
* **Night Mode:** Tuning the color thresholds (HSL) and Canny edge parameters dynamically for low-light driving conditions.
