import cv2
import numpy as np
import math
import os
import torch
from ultralytics import YOLO

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    return cv2.bitwise_and(img, mask)

def select_white_yellow(image):
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower_white = np.uint8([0, 200, 0])
    upper_white = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower_white, upper_white)
    
    lower_yellow = np.uint8([10, 0, 100])
    upper_yellow = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower_yellow, upper_yellow)
    
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=mask)

def draw_lane_lines(img, left_line, right_line, color=[0, 255, 0], thickness=10):
    line_img = np.zeros_like(img)
    poly_pts = np.array([[
        (left_line[0], left_line[1]),
        (left_line[2], left_line[3]),
        (right_line[2], right_line[3]),
        (right_line[0], right_line[1])
    ]], dtype=np.int32)
    cv2.fillPoly(line_img, poly_pts, color)
    return cv2.addWeighted(img, 0.8, line_img, 0.5, 0.0)

# MODIFIED: Pipeline now accepts vehicle bounding boxes to mask them out
def pipeline(image, vehicle_boxes):
    height, width = image.shape[:2]

    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]

    color_filtered_image = select_white_yellow(image)
    
    # --- NEW: DYNAMIC MASKING ---
    # Black out the areas where YOLO found cars so they don't confuse the lane detection
    for box in vehicle_boxes:
        x1, y1, x2, y2 = box
        # Expand the blackout box slightly to cover shadows
        cv2.rectangle(color_filtered_image, (max(0, x1-10), max(0, y1-10)), (min(width, x2+10), min(height, y2+10)), (0, 0, 0), -1)
    # ----------------------------

    gray_image = cv2.cvtColor(color_filtered_image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 50, 150)

    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32)
    )

    lines = cv2.HoughLinesP(
        cropped_image, rho=6, theta=np.pi/180, threshold=50, 
        lines=np.array([]), minLineLength=20, maxLineGap=100
    )

    left_line_x, left_line_y, right_line_x, right_line_y = [], [], [], []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
                if math.fabs(slope) < 0.5:
                    continue
                if slope <= 0:
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])

    min_y = int(height * (3 / 5))
    max_y = height

    left_x_start, left_x_end = 0, int(width/2)
    right_x_start, right_x_end = width, int(width/2)

    if left_line_x and left_line_y:
        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))

    if right_line_x and right_line_y:
        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))

    lane_image = draw_lane_lines(
        image,
        [left_x_start, max_y, left_x_end, min_y],
        [right_x_start, max_y, right_x_end, min_y]
    )

    return lane_image, left_x_start, right_x_start

def estimate_distance(bbox_width):
    focal_length = 800  
    known_width = 1.8   
    if bbox_width == 0: return 999
    return (known_width * focal_length) / bbox_width

def process_video(source=0):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # UPGRADED: Using a smarter model. It will auto-download 'yolov8s.pt' on first run.
    model_path = os.path.join(script_dir, 'yolov8s.pt') 
    model = YOLO(model_path) 
    
    # Enable CUDA for your RTX 4050
    if torch.cuda.is_available():
        print("CUDA detected! Offloading YOLO to RTX 4050...")
        model.to('cuda')
    else:
        print("CUDA not detected. Ensure PyTorch is installed with CUDA support.")

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    # Setup Fullscreen
    cv2.namedWindow('ADAS Project - Live Feed', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('ADAS Project - Live Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (1280, 720))
        frame_center_x = 1280 // 2

        # 1. Run YOLO FIRST (No more frame skipping needed with RTX 4050)
        results = model(resized_frame, classes=[0, 1, 2, 3, 5, 7], conf=0.40, verbose=False)
        
        vehicle_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_boxes.append((x1, y1, x2, y2))

        # 2. Run Lane Detection (Passing the vehicle boxes to mask them out)
        lane_frame, left_lane_x, right_lane_x = pipeline(resized_frame, vehicle_boxes)

        # 3. Tightened Lane Departure Warning
        lane_center = (left_lane_x + right_lane_x) / 2
        drift_offset = frame_center_x - lane_center
        
        # Reduced threshold from 150 to 50 for earlier warnings
        if left_lane_x != 0 and right_lane_x != 1280: 
            if abs(drift_offset) > 50:
                cv2.putText(lane_frame, "WARNING: LANE DEPARTURE!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # 4. Draw YOLO Boxes and Collision Logic
        collision_alert = False
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f'{model.names[cls]} {conf:.2f}'

                bbox_width = x2 - x1
                distance = estimate_distance(bbox_width)

                cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(lane_frame, f"{label} {distance:.1f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                object_center_x = (x1 + x2) / 2
                in_my_lane = left_lane_x < object_center_x < right_lane_x

                if distance < 5.0 and in_my_lane:
                    collision_alert = True
                    cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

        if collision_alert:
            cv2.putText(lane_frame, "COLLISION WARNING!", (400, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        cv2.putText(lane_frame, f"Lane Drift: {drift_offset:.0f}px", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('ADAS Project - Live Feed', lane_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change to 0 for webcam, or put a string path like './highway.mp4' to test video
    process_video(source=0)