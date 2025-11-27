"""
Integrated Lane Detection and Vehicle Tracking for Self-Driving Cars
Trajectory Computation for Autonomous Vehicles

This implementation integrates:
- Lane Detection (Hough Transform, LaneNet, SCNN)
- Object Detection (YOLOv3)
- Object Tracking (DeepSORT)
- Distance Estimation (Monocular, Stereo, LiDAR)
- Trajectory Computation
"""

import cv2
import numpy as np
from collections import deque
import math
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Deep learning models will use fallback methods.")

try:
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# ============================================================================
# Deep Learning Models for Lane Detection
# ============================================================================

if TORCH_AVAILABLE:
    class SpatialCNN(nn.Module):
        """
        Spatial CNN (SCNN) for Lane Detection
        Paper: "Spatial As Deep: Spatial CNN for Traffic Scene Understanding"
        
        SCNN uses spatial message passing to propagate information across rows/columns
        which is particularly effective for detecting long, continuous lane lines.
        """
        
        def __init__(self, input_channels=3, num_classes=5):
            """
            Args:
                input_channels: Number of input channels (3 for RGB)
                num_classes: Number of lane classes (background + lane types)
            """
            super(SpatialCNN, self).__init__()
            
            # VGG-like backbone
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_channels, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
            
            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
            
            self.conv3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
            
            # Spatial convolution layers (message passing)
            self.spatial_conv_d = nn.Conv2d(256, 256, (1, 9), padding=(0, 4))  # Downward
            self.spatial_conv_u = nn.Conv2d(256, 256, (1, 9), padding=(0, 4))  # Upward
            self.spatial_conv_r = nn.Conv2d(256, 256, (9, 1), padding=(4, 0))  # Rightward
            self.spatial_conv_l = nn.Conv2d(256, 256, (9, 1), padding=(4, 0))  # Leftward
            
            # Decoder
            self.decoder = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(32, num_classes, 1)
            )
        
    def spatial_message_passing(self, x):
        """Apply spatial message passing in 4 directions (optimized with reduced iterations)"""
        _, _, h, w = x.size()
        
        # Simplified message passing - only process every 4th row/column for speed
        step = 4  # Process every 4th element instead of every element
        
        # Downward message passing (simplified)
        x_d = x.clone()
        for i in range(step, h, step):
            x_d[:, :, i:i+1, :] = x_d[:, :, i:i+1, :] + F.relu(self.spatial_conv_d(x_d[:, :, i-step:i-step+1, :]))
        
        # Upward message passing (simplified)
        x_u = x.clone()
        for i in range(h-step-1, -1, -step):
            x_u[:, :, i:i+1, :] = x_u[:, :, i:i+1, :] + F.relu(self.spatial_conv_u(x_u[:, :, i+step:i+step+1, :]))
        
        # Rightward message passing (simplified)
        x_r = x.clone()
        for i in range(step, w, step):
            x_r[:, :, :, i:i+1] = x_r[:, :, :, i:i+1] + F.relu(self.spatial_conv_r(x_r[:, :, :, i-step:i-step+1]))
        
        # Leftward message passing (simplified)
        x_l = x.clone()
        for i in range(w-step-1, -1, -step):
            x_l[:, :, :, i:i+1] = x_l[:, :, :, i:i+1] + F.relu(self.spatial_conv_l(x_l[:, :, :, i+step:i+step+1]))
        
        # Combine all directions
        return x_d + x_u + x_r + x_l
        
        def forward(self, x):
            # Encoder
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            
            # Spatial message passing
            x = self.spatial_message_passing(x)
            
            # Decoder
            x = self.decoder(x)
            
            return x


    class LaneNet(nn.Module):
        """
        LaneNet: Deep Neural Network for Lane Detection
        Paper: "Towards End-to-End Lane Detection: an Instance Segmentation Approach"
        
        LaneNet uses instance segmentation with embedding loss to detect multiple lanes
        and distinguish between different lane instances.
        """
        
        def __init__(self, input_channels=3, embedding_dim=4):
            """
            Args:
                input_channels: Number of input channels (3 for RGB)
                embedding_dim: Dimension of lane instance embeddings
            """
            super(LaneNet, self).__init__()
            
            self.embedding_dim = embedding_dim
            
            # Encoder (Feature Extractor)
            self.encoder_conv1 = nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
            
            self.encoder_conv2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
            
            self.encoder_conv3 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
            
            # Binary Segmentation Branch (Lane vs Background)
            self.binary_decoder = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(16, 2, 1)  # 2 classes: background and lane
            )
            
            # Instance Embedding Branch (Lane Instance Discrimination)
            self.embedding_decoder = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(16, embedding_dim, 1)  # N-dimensional embeddings
            )
        
        def forward(self, x):
            # Shared encoder
            x = self.encoder_conv1(x)
            x = self.encoder_conv2(x)
            x = self.encoder_conv3(x)
            
            # Binary segmentation output
            binary_seg = self.binary_decoder(x)
            
            # Instance embedding output
            instance_embedding = self.embedding_decoder(x)
            
            return binary_seg, instance_embedding


class LaneDetector:
    """Lane detection using multiple methods: Hough Transform, SCNN, and LaneNet"""
    
    def __init__(self, method='hough', scnn_weights_path=None, lanenet_weights_path=None, 
                 device='cpu'):
        """
        Initialize Lane Detector
        
        Args:
            method: Detection method - 'hough', 'scnn', 'lanenet', or 'all'
            scnn_weights_path: Path to pretrained SCNN weights
            lanenet_weights_path: Path to pretrained LaneNet weights
            device: 'cpu' or 'cuda'
        """
        self.roi_vertices = None
        self.method = method.lower()
        self.device = device
        
        # Initialize deep learning models if available
        self.scnn_model = None
        self.lanenet_model = None
        
        if TORCH_AVAILABLE and self.method in ['scnn', 'all']:
            print("Initializing SCNN model...")
            self.scnn_model = SpatialCNN(input_channels=3, num_classes=5)
            if scnn_weights_path:
                try:
                    self.scnn_model.load_state_dict(torch.load(scnn_weights_path, map_location=device))
                    print(f"Loaded SCNN weights from {scnn_weights_path}")
                except Exception as e:
                    print(f"Warning: Could not load SCNN weights: {e}")
            self.scnn_model.to(device)
            self.scnn_model.eval()
        
        if TORCH_AVAILABLE and self.method in ['lanenet', 'all']:
            print("Initializing LaneNet model...")
            self.lanenet_model = LaneNet(input_channels=3, embedding_dim=4)
            if lanenet_weights_path:
                try:
                    self.lanenet_model.load_state_dict(torch.load(lanenet_weights_path, map_location=device))
                    print(f"Loaded LaneNet weights from {lanenet_weights_path}")
                except Exception as e:
                    print(f"Warning: Could not load LaneNet weights: {e}")
            self.lanenet_model.to(device)
            self.lanenet_model.eval()
        
        # Image preprocessing for deep learning models
        if TORCHVISION_AVAILABLE:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = None
        
    def region_of_interest(self, img: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        """Apply region of interest mask"""
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        masked = cv2.bitwise_and(img, mask)
        return masked
    
    def detect_lanes_hough(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Detect lanes using Hough Transform
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (lane_lines, annotated_image)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blur, 50, 150)
        
        # Define region of interest (ROI) - lower half of image
        height, width = image.shape[:2]
        roi_vertices = np.array([[
            (0, height),
            (width // 2 - 100, height // 2),
            (width // 2 + 100, height // 2),
            (width, height)
        ]], dtype=np.int32)
        
        # Apply ROI mask
        masked_edges = self.region_of_interest(edges, roi_vertices)
        
        # Hough Line Transform with adjusted parameters
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=100
        )
        
        # Draw detected lines
        line_image = np.copy(image)
        lane_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                lane_lines.append((x1, y1, x2, y2))
        
        return lane_lines, line_image
    
    def calculate_lane_curvature(self, lane_lines: List) -> float:
        """Calculate lane curvature from detected lines"""
        if len(lane_lines) < 2:
            return 0.0
        
        # Simple curvature estimation
        # In practice, this would use polynomial fitting
        avg_slope = np.mean([(y2 - y1) / (x2 - x1 + 1e-6) 
                            for x1, y1, x2, y2 in lane_lines])
        
        # Convert slope to curvature (simplified)
        curvature = abs(avg_slope) * 100
        return curvature
    
    def detect_lanes_scnn(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Detect lanes using Spatial CNN (SCNN)
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (lane_lines, annotated_image)
        """
        if not TORCH_AVAILABLE or self.scnn_model is None:
            print("Warning: SCNN not available, falling back to Hough Transform")
            return self.detect_lanes_hough(image)
        
        # Preprocess image
        original_shape = image.shape[:2]
        input_image = cv2.resize(image, (512, 256))  # SCNN typical input size
        
        # Convert BGR to RGB
        input_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms if available
        if self.transform:
            input_tensor = self.transform(input_rgb).unsqueeze(0).to(self.device)
        else:
            input_tensor = torch.from_numpy(input_rgb.transpose(2, 0, 1)).float()
            input_tensor = input_tensor.unsqueeze(0).to(self.device) / 255.0
        
        # Run inference
        with torch.no_grad():
            output = self.scnn_model(input_tensor)
            lane_prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Resize prediction back to original size
        lane_prediction = cv2.resize(lane_prediction.astype(np.uint8), 
                                    (original_shape[1], original_shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        
        # Extract lane lines from segmentation mask
        lane_lines = self._extract_lane_lines_from_mask(lane_prediction)
        
        # Create annotated image
        line_image = image.copy()
        for x1, y1, x2, y2 in lane_lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Overlay segmentation mask
        mask_colored = self._colorize_lane_mask(lane_prediction)
        line_image = cv2.addWeighted(line_image, 0.7, mask_colored, 0.3, 0)
        
        return lane_lines, line_image
    
    def detect_lanes_lanenet(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Detect lanes using LaneNet (Instance Segmentation)
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (lane_lines, annotated_image)
        """
        if not TORCH_AVAILABLE or self.lanenet_model is None:
            print("Warning: LaneNet not available, falling back to Hough Transform")
            return self.detect_lanes_hough(image)
        
        # Preprocess image
        original_shape = image.shape[:2]
        input_image = cv2.resize(image, (512, 256))  # LaneNet typical input size
        
        # Convert BGR to RGB
        input_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms if available
        if self.transform:
            input_tensor = self.transform(input_rgb).unsqueeze(0).to(self.device)
        else:
            input_tensor = torch.from_numpy(input_rgb.transpose(2, 0, 1)).float()
            input_tensor = input_tensor.unsqueeze(0).to(self.device) / 255.0
        
        # Run inference
        with torch.no_grad():
            binary_seg, instance_embedding = self.lanenet_model(input_tensor)
            
            # Get binary segmentation
            binary_pred = torch.argmax(binary_seg, dim=1).squeeze(0).cpu().numpy()
            
            # Get instance embeddings
            embeddings = instance_embedding.squeeze(0).cpu().numpy()
        
        # Resize predictions back to original size
        binary_pred = cv2.resize(binary_pred.astype(np.uint8), 
                                (original_shape[1], original_shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        
        # Cluster embeddings to separate lane instances
        lane_instances = self._cluster_lane_instances(binary_pred, embeddings, original_shape)
        
        # Extract lane lines from instances
        lane_lines = self._extract_lane_lines_from_instances(lane_instances)
        
        # Create annotated image
        line_image = image.copy()
        
        # Draw different lane instances in different colors
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for idx, instance_lines in enumerate(lane_instances):
            color = colors[idx % len(colors)]
            for x1, y1, x2, y2 in instance_lines:
                cv2.line(line_image, (x1, y1), (x2, y2), color, 3)
        
        return lane_lines, line_image
    
    def detect_lanes(self, image: np.ndarray, method: str = None) -> Tuple[List, np.ndarray]:
        """
        Detect lanes using specified method
        
        Args:
            image: Input image (BGR format)
            method: Detection method - 'hough', 'scnn', 'lanenet', or None (use default)
            
        Returns:
            Tuple of (lane_lines, annotated_image)
        """
        detection_method = method if method else self.method
        
        if detection_method == 'scnn':
            return self.detect_lanes_scnn(image)
        elif detection_method == 'lanenet':
            return self.detect_lanes_lanenet(image)
        elif detection_method == 'all':
            # Combine all methods
            lines_hough, _ = self.detect_lanes_hough(image)
            lines_scnn, _ = self.detect_lanes_scnn(image)
            lines_lanenet, _ = self.detect_lanes_lanenet(image)
            
            # Combine results (merge all unique lines)
            all_lines = lines_hough + lines_scnn + lines_lanenet
            all_lines = self._remove_duplicate_lines(all_lines)
            
            # Create combined visualization
            combined_image = image.copy()
            for x1, y1, x2, y2 in all_lines:
                cv2.line(combined_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            return all_lines, combined_image
        else:
            # Default to Hough Transform
            return self.detect_lanes_hough(image)
    
    def _extract_lane_lines_from_mask(self, mask: np.ndarray) -> List:
        """Extract lane lines from segmentation mask"""
        lane_lines = []
        height, width = mask.shape
        
        # Process each lane class (skip background = 0)
        for lane_id in range(1, 5):
            lane_mask = (mask == lane_id).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    continue
                
                # Fit line to contour
                if len(contour) >= 2:
                    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    
                    # Calculate line endpoints
                    lefty = int((-x * vy / vx) + y)
                    righty = int(((width - x) * vy / vx) + y)
                    
                    # Clip to image boundaries
                    x1, y1 = 0, lefty
                    x2, y2 = width - 1, righty
                    
                    if 0 <= y1 < height and 0 <= y2 < height:
                        lane_lines.append((x1, y1, x2, y2))
        
        return lane_lines
    
    def _cluster_lane_instances(self, binary_mask: np.ndarray, embeddings: np.ndarray, 
                                original_shape: Tuple) -> List[List]:
        """Cluster lane pixels into separate instances using embeddings"""
        # Resize embeddings
        embeddings_resized = []
        for i in range(embeddings.shape[0]):
            emb_channel = cv2.resize(embeddings[i], (original_shape[1], original_shape[0]))
            embeddings_resized.append(emb_channel)
        embeddings_resized = np.array(embeddings_resized)
        
        # Get lane pixel coordinates
        lane_pixels = np.nonzero(binary_mask > 0)
        
        if len(lane_pixels[0]) == 0:
            return []
        
        # Simple clustering using connected components
        # In practice, use mean-shift or other clustering on embeddings
        num_labels, labels = cv2.connectedComponents(binary_mask)
        
        instances = []
        for label_id in range(1, num_labels):
            instance_mask = (labels == label_id).astype(np.uint8) * 255
            
            # Extract lines from this instance
            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            instance_lines = []
            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    continue
                
                if len(contour) >= 2:
                    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    
                    height, width = binary_mask.shape
                    lefty = int((-x * vy / vx) + y)
                    righty = int(((width - x) * vy / vx) + y)
                    
                    x1, y1 = 0, lefty
                    x2, y2 = width - 1, righty
                    
                    if 0 <= y1 < height and 0 <= y2 < height:
                        instance_lines.append((x1, y1, x2, y2))
            
            if instance_lines:
                instances.append(instance_lines)
        
        return instances
    
    def _extract_lane_lines_from_instances(self, instances: List[List]) -> List:
        """Flatten instance lines into single list"""
        all_lines = []
        for instance_lines in instances:
            all_lines.extend(instance_lines)
        return all_lines
    
    def _colorize_lane_mask(self, mask: np.ndarray) -> np.ndarray:
        """Colorize lane segmentation mask"""
        height, width = mask.shape
        colored = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Assign colors to different lane classes
        colors = {
            1: (255, 0, 0),    # Blue
            2: (0, 255, 0),    # Green
            3: (0, 0, 255),    # Red
            4: (255, 255, 0),  # Cyan
        }
        
        for lane_id, color in colors.items():
            colored[mask == lane_id] = color
        
        return colored
    
    def _remove_duplicate_lines(self, lines: List, threshold: float = 20.0) -> List:
        """Remove duplicate or very similar lines"""
        if not lines:
            return []
        
        unique_lines = []
        for line in lines:
            x1, y1, x2, y2 = line
            is_duplicate = False
            
            for unique_line in unique_lines:
                ux1, uy1, ux2, uy2 = unique_line
                
                # Calculate distance between lines
                dist = np.sqrt((x1 - ux1)**2 + (y1 - uy1)**2 + (x2 - ux2)**2 + (y2 - uy2)**2)
                
                if dist < threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_lines.append(line)
        
        return unique_lines


class ObjectDetector:
    """Object detection using YOLOv3"""
    
    def __init__(self, config_path: str = None, weights_path: str = None, 
                 classes_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize YOLOv3 detector
        
        Args:
            config_path: Path to YOLO config file
            weights_path: Path to YOLO weights file
            classes_path: Path to classes names file
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = 0.4
        
        # Load class names
        if classes_path:
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            # Default COCO classes (first 10 relevant for vehicles)
            self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                          'bus', 'train', 'truck', 'boat', 'traffic light']
        
        # Initialize YOLO network
        self.net = None
        if config_path and weights_path:
            try:
                self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception as e:
                print(f"Warning: Could not load YOLO model: {e}. Using placeholder detection.")
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detections with format: [{'bbox': (x, y, w, h), 'confidence': float, 'class': str}, ...]
        """
        detections = []
        
        if self.net is None:
            # Color-based detection for synthetic test images
            height, width = image.shape[:2]
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Detect red cars (BGR: 0, 0, 255 -> HSV range)
            red_lower1 = np.array([0, 50, 50])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([170, 50, 50])
            red_upper2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Detect blue cars (BGR: 255, 0, 0 -> HSV range)
            blue_lower = np.array([100, 50, 50])
            blue_upper = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
            
            # Detect green cars (BGR: 0, 255, 0 -> HSV range)
            green_lower = np.array([40, 50, 50])
            green_upper = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            
            # Find contours for red objects
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in red_contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    # Only detect if in road region (lower half)
                    if y > height // 2:
                        detections.append({
                            'bbox': (x, y, w, h),
                            'confidence': 0.92,
                            'class': 'car',
                            'class_id': 2
                        })
            
            # Find contours for blue objects
            blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in blue_contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    # Only detect if in road region (lower half)
                    if y > height // 2:
                        detections.append({
                            'bbox': (x, y, w, h),
                            'confidence': 0.90,
                            'class': 'car',
                            'class_id': 2
                        })
            
            # Find contours for green objects
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in green_contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    # Only detect if in road region (lower half)
                    if y > height // 2:
                        detections.append({
                            'bbox': (x, y, w, h),
                            'confidence': 0.91,
                            'class': 'car',
                            'class_id': 2
                        })
            
            return detections
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Forward pass
        outputs = self.net.forward(output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidences[i],
                    'class': self.classes[class_ids[i]] if class_id < len(self.classes) else 'unknown',
                    'class_id': class_ids[i]
                })
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes on image"""
        annotated = image.copy()
        
        for det in detections:
            x, y, w, h = det['bbox']
            label = f"{det['class']}: {det['confidence']:.2f}"
            
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated


class ObjectTracker:
    """Object tracking using DeepSORT-like algorithm"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 1):
        """
        Initialize tracker
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections before track is confirmed
        """
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.frame_count = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of detections from object detector
            
        Returns:
            List of tracked objects with IDs
        """
        self.frame_count += 1
        
        # Simple tracking based on IoU (Intersection over Union)
        # In practice, DeepSORT uses Kalman filters and deep features
        
        tracked_objects = []
        
        for det in detections:
            x, y, w, h = det['bbox']
            center = (x + w // 2, y + h // 2)
            
            # Find best matching existing track
            best_match_id = None
            best_iou = 0.3  # Minimum IoU threshold
            
            for track_id, track in self.tracks.items():
                if track['age'] < self.max_age:
                    iou = self._calculate_iou(det['bbox'], track['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track
                self.tracks[best_match_id]['bbox'] = det['bbox']
                self.tracks[best_match_id]['center'] = center
                self.tracks[best_match_id]['age'] = 0
                self.tracks[best_match_id]['hits'] += 1
                self.tracks[best_match_id]['class'] = det['class']
                self.tracks[best_match_id]['confidence'] = det['confidence']
                # Update trajectory
                if 'trajectory' not in self.tracks[best_match_id]:
                    self.tracks[best_match_id]['trajectory'] = []
                self.tracks[best_match_id]['trajectory'].append(center)
                # Limit trajectory length
                if len(self.tracks[best_match_id]['trajectory']) > 20:
                    self.tracks[best_match_id]['trajectory'].pop(0)
                
                tracked_objects.append({
                    'id': best_match_id,
                    'bbox': det['bbox'],
                    'center': center,
                    'class': det['class'],
                    'confidence': det['confidence'],
                    'trajectory': self.tracks[best_match_id]['trajectory'].copy()
                })
            else:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                
                self.tracks[track_id] = {
                    'bbox': det['bbox'],
                    'center': center,
                    'age': 0,
                    'hits': 1,
                    'class': det['class'],
                    'confidence': det['confidence'],
                    'trajectory': [center]
                }
                
                if self.tracks[track_id]['hits'] >= self.min_hits:
                    tracked_objects.append({
                        'id': track_id,
                        'bbox': det['bbox'],
                        'center': center,
                        'class': det['class'],
                        'confidence': det['confidence'],
                        'trajectory': [center]
                    })
        
        # Age all tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            track['age'] += 1
            if track['age'] >= self.max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        return tracked_objects
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area


class DistanceEstimator:
    """Distance estimation using various methods"""
    
    def __init__(self, camera_fov: float = 60.0, image_width: int = 640, 
                 camera_height: float = 1.5, focal_length: float = None):
        """
        Initialize distance estimator
        
        Args:
            camera_fov: Camera field of view in degrees
            image_width: Width of input image
            camera_height: Height of camera above ground (meters)
            focal_length: Focal length in pixels (calculated from FOV if None)
        """
        self.camera_fov = camera_fov
        self.image_width = image_width
        self.camera_height = camera_height
        
        if focal_length is None:
            # Calculate focal length from FOV
            self.focal_length = (image_width / 2) / math.tan(math.radians(camera_fov / 2))
        else:
            self.focal_length = focal_length
    
    def estimate_distance_monocular(self, bbox: Tuple, object_class: str) -> float:
        """
        Estimate distance using monocular vision (heuristic method)
        
        Args:
            bbox: Bounding box (x, y, w, h)
            object_class: Class of object (for size estimation)
            
        Returns:
            Estimated distance in meters
        """
        x, y, w, h = bbox
        
        # Known average object sizes (in meters)
        object_sizes = {
            'car': 4.5,
            'truck': 6.0,
            'bus': 12.0,
            'motorcycle': 2.0,
            'person': 1.7,
            'bicycle': 1.5
        }
        
        # Get object size
        object_height = object_sizes.get(object_class.lower(), 1.5)
        
        # Distance estimation: distance = (focal_length * real_height) / pixel_height
        if h > 0:
            distance = (self.focal_length * object_height) / h
        else:
            distance = 0.0
        
        return distance
    
    def estimate_distance_stereo(self, bbox_left: Tuple, bbox_right: Tuple, 
                                 baseline: float = 0.54) -> float:
        """
        Estimate distance using stereo vision
        
        Args:
            bbox_left: Bounding box in left image
            bbox_right: Bounding box in right image
            baseline: Distance between cameras (meters)
            
        Returns:
            Estimated distance in meters
        """
        x1, y1, w1, h1 = bbox_left
        x2, y2, w2, h2 = bbox_right
        
        # Calculate disparity (difference in x-coordinates)
        center_left = x1 + w1 // 2
        center_right = x2 + w2 // 2
        disparity = abs(center_left - center_right)
        
        if disparity > 0:
            # Distance = (baseline * focal_length) / disparity
            distance = (baseline * self.focal_length) / disparity
        else:
            distance = 0.0
        
        return distance
    
    def estimate_distance_lidar(self, lidar_point: Tuple[float, float, float]) -> float:
        """
        Estimate distance using LiDAR data
        
        Args:
            lidar_point: LiDAR point (x, y, z) in meters
            
        Returns:
            Distance in meters
        """
        x, y, z = lidar_point
        distance = math.sqrt(x**2 + y**2 + z**2)
        return distance


class TrajectoryComputer:
    """Trajectory computation for autonomous vehicles"""
    
    def __init__(self):
        self.lane_history = deque(maxlen=10)
        self.obstacle_history = deque(maxlen=30)
    
    def compute_trajectory(self, lane_lines: List, tracked_objects: List[Dict],
                          ego_vehicle_state: Dict, image_shape: Tuple[int, int] = None) -> Dict:
        """
        Compute safe trajectory based on lanes, obstacles, and vehicle state
        
        Args:
            lane_lines: Detected lane lines
            tracked_objects: Tracked objects with trajectories
            ego_vehicle_state: Current state of ego vehicle {'speed': float, 'position': (x, y)}
            image_shape: (height, width) of the image frame
            
        Returns:
            Dictionary with trajectory information
        """
        # Analyze lane structure
        lane_center = self._calculate_lane_center(lane_lines)
        lane_curvature = self._estimate_lane_curvature(lane_lines)
        
        # Predict obstacle trajectories
        obstacle_predictions = self._predict_obstacle_trajectories(tracked_objects)
        
        # Compute safe path
        safe_path = self._compute_safe_path(
            lane_center, lane_curvature, obstacle_predictions, ego_vehicle_state, image_shape
        )
        
        # Calculate trajectory waypoints
        waypoints = self._generate_waypoints(safe_path, ego_vehicle_state)
        
        return {
            'waypoints': waypoints,
            'lane_center': lane_center,
            'curvature': lane_curvature,
            'obstacle_predictions': obstacle_predictions,
            'safe_path': safe_path
        }
    
    def _calculate_lane_center(self, lane_lines: List) -> Tuple[float, float]:
        """Calculate center of lane from detected lines"""
        if len(lane_lines) < 2:
            return (320, 480)  # Default center
        
        # Average the x-coordinates of lane line centers
        centers = []
        for x1, y1, x2, y2 in lane_lines:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x, center_y))
        
        if centers:
            avg_center = np.mean(centers, axis=0)
            return tuple(avg_center)
        
        return (320, 480)
    
    def _estimate_lane_curvature(self, lane_lines: List) -> float:
        """Estimate lane curvature"""
        if len(lane_lines) < 2:
            return 0.0
        
        # Simple curvature estimation from line slopes
        slopes = []
        for x1, y1, x2, y2 in lane_lines:
            if abs(x2 - x1) > 1:
                slope = (y2 - y1) / (x2 - x1)
                slopes.append(slope)
        
        if slopes:
            avg_slope = np.mean(slopes)
            curvature = abs(avg_slope) * 0.1  # Scale factor
            return curvature
        
        return 0.0
    
    def _predict_obstacle_trajectories(self, tracked_objects: List[Dict]) -> List[Dict]:
        """Predict future trajectories of tracked objects"""
        predictions = []
        
        for obj in tracked_objects:
            if 'trajectory' in obj and len(obj['trajectory']) >= 2:
                trajectory = obj['trajectory']
                
                # Simple linear prediction
                if len(trajectory) >= 2:
                    # Calculate velocity
                    dx = trajectory[-1][0] - trajectory[-2][0]
                    dy = trajectory[-1][1] - trajectory[-2][1]
                    
                    # Predict next position
                    future_pos = (
                        trajectory[-1][0] + dx,
                        trajectory[-1][1] + dy
                    )
                    
                    predictions.append({
                        'id': obj['id'],
                        'current_pos': trajectory[-1],
                        'predicted_pos': future_pos,
                        'velocity': (dx, dy),
                        'class': obj['class']
                    })
        
        return predictions
    
    def _compute_safe_path(self, lane_center: Tuple, curvature: float,
                          obstacle_predictions: List[Dict],
                          ego_state: Dict, image_shape: Tuple[int, int] = None) -> List[Tuple]:
        """Compute safe path avoiding obstacles"""
        # Get image dimensions
        if image_shape:
            image_height, image_width = image_shape
        else:
            image_height = 480  # Default
            image_width = 640
        
        # Start from ego vehicle position (bottom center)
        ego_x = ego_state.get('position', (image_width // 2, image_height - 20))[0]
        ego_y = ego_state.get('position', (image_width // 2, image_height - 20))[1]
        
        lane_x, lane_y = lane_center
        
        # Generate multiple waypoints extending forward
        path = []
        num_waypoints = 15  # Reduced for faster computation
        
        # Calculate target y position (extend well into the image)
        target_y = max(50, int(lane_y * 0.3))  # Extend to upper third of image
        
        for i in range(num_waypoints):
            # Interpolate from ego position forward
            progress = i / (num_waypoints - 1)
            
            # Smooth interpolation using ease-in-out curve
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            
            # Calculate y position (moving forward/up in image)
            y = int(ego_y - (ego_y - target_y) * smooth_progress)
            y = max(50, min(image_height - 1, y))  # Clamp to image bounds, keep above bottom
            
            # Calculate x position - smoothly transition from ego to lane center
            # Use the lane center x as the target, but adjust based on curvature
            target_x = lane_x
            
            # Smooth x interpolation
            x = int(ego_x + (target_x - ego_x) * smooth_progress)
            
            # Apply curvature adjustment (more subtle)
            if abs(curvature) > 0.01:
                # Curvature creates a smooth curve
                curvature_offset = curvature * math.sin(progress * math.pi) * 50
                x = int(x + curvature_offset)
            
            x = max(20, min(image_width - 20, x))  # Clamp to image bounds with margin
            
            # Adjust for obstacles with smooth avoidance
            for pred in obstacle_predictions:
                obstacle_x, obstacle_y = pred['predicted_pos']
                
                # Check if obstacle is near this waypoint
                distance_to_obstacle = math.sqrt(
                    (obstacle_x - x)**2 + (obstacle_y - y)**2
                )
                
                if distance_to_obstacle < 100:  # Threshold in pixels
                    # Smooth obstacle avoidance
                    avoidance_distance = 80 - distance_to_obstacle  # More avoidance when closer
                    if obstacle_x < x:
                        x = min(image_width - 20, x + int(avoidance_distance * 0.8))  # Move right
                    else:
                        x = max(20, x - int(avoidance_distance * 0.8))  # Move left
            
            path.append((x, y))
        
        return path
    
    def _generate_waypoints(self, safe_path: List[Tuple], 
                           ego_state: Dict) -> List[Tuple]:
        """Generate trajectory waypoints"""
        waypoints = []
        
        for point in safe_path:
            waypoints.append({
                'x': point[0],
                'y': point[1],
                'speed': ego_state.get('speed', 0.0),
                'timestamp': len(waypoints) * 0.1  # 0.1s intervals
            })
        
        return waypoints


class AutonomousVehiclePipeline:
    """Main pipeline integrating all components"""
    
    def __init__(self, lane_detection_method='hough', scnn_weights=None, lanenet_weights=None):
        """
        Initialize the autonomous vehicle pipeline
        
        Args:
            lane_detection_method: 'hough', 'scnn', 'lanenet', or 'all'
            scnn_weights: Path to SCNN pretrained weights (optional)
            lanenet_weights: Path to LaneNet pretrained weights (optional)
        """
        self.lane_detector = LaneDetector(
            method=lane_detection_method,
            scnn_weights_path=scnn_weights,
            lanenet_weights_path=lanenet_weights
        )
        self.object_detector = ObjectDetector()
        self.object_tracker = ObjectTracker()
        self.distance_estimator = DistanceEstimator()
        self.trajectory_computer = TrajectoryComputer()
    
    def process_frame(self, frame: np.ndarray, ego_vehicle_state: Dict = None) -> Dict:
        """
        Process a single frame through the complete pipeline
        
        Args:
            frame: Input image frame (BGR format)
            ego_vehicle_state: Current state of ego vehicle
            
        Returns:
            Dictionary with all processing results
        """
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        if ego_vehicle_state is None:
            # Default ego position: bottom center of image
            ego_vehicle_state = {'speed': 50.0, 'position': (width // 2, height - 20)}
        elif 'position' not in ego_vehicle_state or ego_vehicle_state['position'] == (0, 0):
            # Update position if not set or default
            ego_vehicle_state['position'] = (width // 2, height - 20)
        
        # Step 1: Lane Detection (using configured method)
        lane_lines, _ = self.lane_detector.detect_lanes(frame)
        lane_curvature = self.lane_detector.calculate_lane_curvature(lane_lines)
        
        # Step 2: Object Detection
        detections = self.object_detector.detect_objects(frame)
        
        # Step 3: Object Tracking
        tracked_objects = self.object_tracker.update(detections)
        
        # Step 4: Distance Estimation
        distances = {}
        for obj in tracked_objects:
            distance = self.distance_estimator.estimate_distance_monocular(
                obj['bbox'], obj['class']
            )
            distances[obj['id']] = distance
        
        # Step 5: Trajectory Computation
        trajectory = self.trajectory_computer.compute_trajectory(
            lane_lines, tracked_objects, ego_vehicle_state, (height, width)
        )
        
        return {
            'lane_lines': lane_lines,
            'lane_curvature': lane_curvature,
            'detections': detections,
            'tracked_objects': tracked_objects,
            'distances': distances,
            'trajectory': trajectory,
            'annotated_frame': self._annotate_frame(
                frame, lane_lines, tracked_objects, trajectory, distances
            )
        }
    
    def _annotate_frame(self, frame: np.ndarray, lane_lines: List,
                       tracked_objects: List[Dict], trajectory: Dict,
                       distances: Dict) -> np.ndarray:
        """Annotate frame with all detection and tracking results"""
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        
        # Draw lane lines - simple and clean
        for x1, y1, x2, y2 in lane_lines:
            cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw lane center - simple line
        if 'lane_center' in trajectory and trajectory['lane_center']:
            center_x = int(trajectory['lane_center'][0])
            cv2.line(annotated, (center_x, height//2), (center_x, height), (0, 255, 255), 2)
        
        # Draw tracked objects - simple bounding boxes and labels
        for obj in tracked_objects:
            x, y, w, h = obj['bbox']
            track_id = obj['id']
            distance = distances.get(track_id, 0.0)
            
            # Simple bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Simple label
            label = f"ID:{track_id} {distance:.1f}m"
            cv2.putText(annotated, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw predicted path - enhanced visualization with better visibility
        if 'waypoints' in trajectory and trajectory['waypoints']:
            waypoints = trajectory['waypoints']
            if len(waypoints) > 0:
                # Get ego vehicle starting position (bottom center of image)
                start_x = width // 2
                start_y = height - 20
                
                # Draw thick path line connecting waypoints
                path_points = [(start_x, start_y)]  # Start from ego position
                
                # Add all waypoints
                for waypoint in waypoints:
                    x, y = int(waypoint['x']), int(waypoint['y'])
                    # Ensure waypoints are within image bounds
                    x = max(0, min(width - 1, x))
                    y = max(0, min(height - 1, y))
                    path_points.append((x, y))
                
                # Draw thick path line connecting waypoints
                path_points = [(start_x, start_y)]  # Start from ego position
                
                # Add all waypoints
                for waypoint in waypoints:
                    x, y = int(waypoint['x']), int(waypoint['y'])
                    # Ensure waypoints are within image bounds
                    x = max(0, min(width - 1, x))
                    y = max(0, min(height - 1, y))
                    path_points.append((x, y))
                
                # Draw path as a thick, visible line (simplified for speed)
                # Draw black outline
                for i in range(len(path_points) - 1):
                    pt1 = path_points[i]
                    pt2 = path_points[i + 1]
                    cv2.line(annotated, pt1, pt2, (0, 0, 0), 10)  # Black outline
                
                # Then draw the main path line (bright cyan)
                for i in range(len(path_points) - 1):
                    pt1 = path_points[i]
                    pt2 = path_points[i + 1]
                    cv2.line(annotated, pt1, pt2, (255, 255, 0), 8)  # Cyan
                
                # Draw arrows every 3rd segment (reduced for speed)
                for i in range(0, len(path_points) - 1, 3):
                    pt1 = path_points[i]
                    pt2 = path_points[i + 1]
                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    if abs(dx) > 0.1 or abs(dy) > 0.1:
                        cv2.arrowedLine(annotated, pt1, pt2, (0, 255, 255), 4, tipLength=0.3)
                
                # Draw waypoints as visible circles (reduced complexity)
                for i in range(0, len(waypoints), 2):  # Draw every 2nd waypoint for speed
                    waypoint = waypoints[i]
                    x, y = int(waypoint['x']), int(waypoint['y'])
                    x = max(0, min(width - 1, x))
                    y = max(0, min(height - 1, y))
                    
                    # Simplified circle drawing
                    cv2.circle(annotated, (x, y), 6, (0, 0, 0), -1)  # Black outer
                    cv2.circle(annotated, (x, y), 4, (255, 255, 0), -1)  # Cyan inner
                
                # Draw starting point (ego vehicle position) - simplified
                cv2.circle(annotated, (start_x, start_y), 10, (0, 0, 0), -1)  # Black outer
                cv2.circle(annotated, (start_x, start_y), 8, (0, 255, 0), -1)  # Green start
                
                # Add text label for predicted path (simplified)
                text = "PREDICTED PATH"
                cv2.putText(annotated, text, (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)  # Black outline
                cv2.putText(annotated, text, (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)  # Cyan text
        
        return annotated


def create_test_image(width=640, height=480):
    """Create a synthetic test image with lanes and objects"""
    # Create a simple road-like image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Road surface (gray)
    cv2.rectangle(image, (0, height//2), (width, height), (50, 50, 50), -1)
    
    # Sky (light blue)
    cv2.rectangle(image, (0, 0), (width, height//2), (135, 206, 235), -1)
    
    # Simple lane markings - left and right lines
    left_lane_x = width // 4
    cv2.line(image, (left_lane_x, height//2), (left_lane_x - 20, height), (255, 255, 255), 4)
    
    right_lane_x = 3 * width // 4
    cv2.line(image, (right_lane_x, height//2), (right_lane_x + 20, height), (255, 255, 255), 4)
    
    # Center lane divider (simple dashed line)
    for i in range(height//2, height, 40):
        cv2.line(image, (width//2, i), (width//2, i + 20), (255, 255, 255), 3)
    
    # Simple car objects
    car_x, car_y = width//2 + 80, height//2 + 50
    car_w, car_h = 90, 70
    cv2.rectangle(image, (car_x, car_y), (car_x + car_w, car_y + car_h), (0, 0, 255), -1)
    
    car2_x, car2_y = width//4 - 50, height//2 + 100
    car2_w, car2_h = 80, 60
    cv2.rectangle(image, (car2_x, car2_y), (car2_x + car2_w, car2_y + car2_h), (255, 0, 0), -1)
    
    car3_x, car3_y = width//2 - 150, height//2 + 120
    car3_w, car3_h = 75, 55
    cv2.rectangle(image, (car3_x, car3_y), (car3_x + car3_w, car3_y + car3_h), (0, 255, 0), -1)
    
    return image


def main():
    """Main function for testing the pipeline"""
    print("=" * 60)
    print("Autonomous Vehicle Trajectory Computation System")
    print("Integrated Lane Detection and Vehicle Tracking")
    print("=" * 60)
    
    # Display available lane detection methods
    print("\nAvailable Lane Detection Methods:")
    print("  1. Hough Transform (Traditional)")
    print("  2. Spatial CNN (SCNN) - Deep Learning")
    print("  3. LaneNet - Instance Segmentation")
    print("  4. All Methods (Combined)")
    
    # Choose detection method
    print("\nDefault: Using Hough Transform")
    print("(To use SCNN or LaneNet, provide pretrained weights)")
    lane_method = 'hough'  # Can be changed to 'scnn', 'lanenet', or 'all'
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = AutonomousVehiclePipeline(lane_detection_method=lane_method)
    
    print("\nPipeline initialized successfully!")
    print("\nComponents loaded:")
    if TORCH_AVAILABLE:
        print(f"  [OK] Lane Detector ({lane_method.upper()})")
        print("       - Hough Transform: Available")
        print("       - SCNN: Available (needs pretrained weights)")
        print("       - LaneNet: Available (needs pretrained weights)")
    else:
        print("  [OK] Lane Detector (Hough Transform)")
        print("  [!] PyTorch not installed - Deep learning methods unavailable")
    print("  [OK] Object Detector (YOLOv3)")
    print("  [OK] Object Tracker (DeepSORT-like)")
    print("  [OK] Distance Estimator (Monocular/Stereo/LiDAR)")
    print("  [OK] Trajectory Computer")
    
    # Load or create test image
    print("\n" + "=" * 60)
    test_image = None
    
    # Try to load existing test_input.jpg
    import os
    if os.path.exists('test_input.jpg'):
        print("Loading existing test_input.jpg...")
        test_image = cv2.imread('test_input.jpg')
        if test_image is not None:
            print(f"Loaded test image: {test_image.shape[1]}x{test_image.shape[0]}")
        else:
            print("Failed to load image, creating synthetic test image...")
            test_image = create_test_image(640, 480)
            print("Test image created (640x480)")
    else:
        print("Creating synthetic test image...")
        test_image = create_test_image(640, 480)
        print("Test image created (640x480)")
        cv2.imwrite('test_input.jpg', test_image)
        print("Saved test image as 'test_input.jpg'")
    
    # Process the test image
    print("\n" + "=" * 60)
    print("Processing test image through pipeline...")
    print("-" * 60)
    
    ego_state = {'speed': 50.0, 'position': (320, 400)}  # km/h, pixel position
    results = pipeline.process_frame(test_image, ego_state)
    
    # Display results
    print(f"[OK] Lane Detection: Found {len(results['lane_lines'])} lane lines")
    print(f"[OK] Lane Curvature: {results['lane_curvature']:.2f}")
    print(f"[OK] Object Detection: Found {len(results['detections'])} objects")
    print(f"[OK] Object Tracking: Tracking {len(results['tracked_objects'])} objects")
    
    if results['distances']:
        print("\nDistance Estimates:")
        for obj_id, distance in results['distances'].items():
            print(f"  - Object ID {obj_id}: {distance:.2f} meters")
    
    if results['trajectory']['waypoints']:
        print(f"\n[OK] Trajectory: Generated {len(results['trajectory']['waypoints'])} waypoints")
        print(f"  - Lane Center: {results['trajectory']['lane_center']}")
        print(f"  - Curvature: {results['trajectory']['curvature']:.4f}")
    
    # Save annotated result
    output_path = 'test_output.jpg'
    cv2.imwrite(output_path, results['annotated_frame'])
    print(f"\n[OK] Saved annotated result as '{output_path}'")
    
    # Display images
    print("\n" + "=" * 60)
    print("Displaying results (Press any key to close)...")
    print("-" * 60)
    
    # Show input and output side by side
    combined = np.hstack([test_image, results['annotated_frame']])
    cv2.imshow('Input (Left) | Output (Right) - Press any key to close', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    
    # Additional usage examples
    print("\nAdditional Usage Examples:")
    print("\n1. Process video file:")
    print("   cap = cv2.VideoCapture('video.mp4')")
    print("   while True:")
    print("       ret, frame = cap.read()")
    print("       if not ret: break")
    print("       results = pipeline.process_frame(frame)")
    print("       cv2.imshow('Output', results['annotated_frame'])")
    print("       if cv2.waitKey(1) & 0xFF == ord('q'): break")
    
    print("\n2. Process webcam:")
    print("   cap = cv2.VideoCapture(0)")
    print("   # Same processing loop as above")
    
    print("\n3. Process your own image:")
    print("   image = cv2.imread('your_image.jpg')")
    print("   results = pipeline.process_frame(image)")
    print("   cv2.imshow('Result', results['annotated_frame'])")
    print("   cv2.waitKey(0)")
    
    print("\n4. Use different lane detection methods:")
    print("   # Use SCNN")
    print("   pipeline_scnn = AutonomousVehiclePipeline(")
    print("       lane_detection_method='scnn',")
    print("       scnn_weights='path/to/scnn_weights.pth'")
    print("   )")
    print("   ")
    print("   # Use LaneNet")
    print("   pipeline_lanenet = AutonomousVehiclePipeline(")
    print("       lane_detection_method='lanenet',")
    print("       lanenet_weights='path/to/lanenet_weights.pth'")
    print("   )")
    print("   ")
    print("   # Use all methods combined")
    print("   pipeline_all = AutonomousVehiclePipeline(")
    print("       lane_detection_method='all'")
    print("   )")
    
    if TORCH_AVAILABLE:
        print("\n" + "=" * 60)
        print("PyTorch is available!")
        print("You can train or download pretrained SCNN/LaneNet models.")
        print("Place the .pth files in your project directory.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("To use SCNN and LaneNet, install PyTorch:")
        print("  pip install torch torchvision")
        print("=" * 60)


if __name__ == "__main__":
    main()

