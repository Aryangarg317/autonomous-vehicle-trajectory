import os
import cv2
import numpy as np
import math
import time
import sys
from ultralytics import YOLO  # YOLOv8 module

# Function to mask out the region of interest
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    return cv2.bitwise_and(img, mask)

def select_white_yellow(image):
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # White Color Mask
    lower_white = np.uint8([0, 200, 0])
    upper_white = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower_white, upper_white)
    # Yellow Color Mask
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

# MODIFIED: Pipeline now returns lane boundary X-coordinates at the bottom of the screen
def pipeline(image):
    height, width = image.shape[:2]

    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]

    color_filtered_image = select_white_yellow(image)
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

    # Default values if lanes aren't found
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

    # Return the processed image AND the bottom x-coordinates for the left and right lanes
    return lane_image, left_x_start, right_x_start

def estimate_distance(bbox_width):
    focal_length = 800  # Tuned for standard webcams
    known_width = 1.8   # Average car width in meters
    if bbox_width == 0: return 999
    return (known_width * focal_length) / bbox_width
    
# MODIFIED: Changed to default to webcam (0)
def process_video(source=0):
    # Lock down the exact file path so YOLO stops downloading
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'yolov8n.pt')
    
    # Load YOLO using that strict path
    model = YOLO(model_path) 
    
    # --- THIS WAS THE MISSING LINE ---
    # Open the webcam using the source passed in the arguments
    cap = cv2.VideoCapture(source)
    # ---------------------------------

    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # ... the rest of the while loop continues here ...

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (1280, 720))
        frame_center_x = 1280 // 2

        # 1. Get Lane Image and Boundaries
        lane_frame, left_lane_x, right_lane_x = pipeline(resized_frame)

        # 2. Lane Departure Warning Logic
        lane_center = (left_lane_x + right_lane_x) / 2
        drift_offset = frame_center_x - lane_center
        
        # If the car is drifting too far left or right (Threshold: 150 pixels)
        if left_lane_x != 0 and right_lane_x != 1280: # Ensure lanes are actually detected
            if abs(drift_offset) > 150:
                cv2.putText(lane_frame, "WARNING: LANE DEPARTURE!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # 3. Object Detection & Forward Collision Logic
        results = model(resized_frame, classes=[0, 1, 2, 3, 5, 7], conf=0.50, verbose=False)
        
        collision_alert = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f'{model.names[cls]} {conf:.2f}'

                bbox_width = x2 - x1
                distance = estimate_distance(bbox_width)

                # Draw standard bounding box
                cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(lane_frame, f"{label} {distance:.1f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Forward Collision / Sudden Incursion Logic
                # Check if object is close AND horizontally within our lane bounds
                object_center_x = (x1 + x2) / 2
                in_my_lane = left_lane_x < object_center_x < right_lane_x

                if distance < 5.0 and in_my_lane:
                    collision_alert = True
                    # Turn bounding box red
                    cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

        if collision_alert:
            cv2.putText(lane_frame, "COLLISION WARNING!", (400, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        # Draw ADAS Dashboard Data
        cv2.putText(lane_frame, f"Lane Drift: {drift_offset:.0f}px", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('ADAS Project - Live Feed', lane_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video(source=1)