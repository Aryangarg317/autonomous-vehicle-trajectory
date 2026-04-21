import cv2
import numpy as np
import math
import os
import torch
import time
import threading
import winsound
from ultralytics import YOLO

# ==========================================
# AUDIO ALERT SYSTEM
# ==========================================
# ==========================================
# AUDIO ALERT SYSTEM (3 Real-World Profiles)
# ==========================================
last_beep_time = 0
def play_alert(alert_type):
    if alert_type == 'ldw':
        # Profile: Rumble Strips
        # 3 rapid, low-frequency bursts to simulate driving over grooved pavement
        winsound.Beep(350, 100)
        time.sleep(0.05)
        winsound.Beep(350, 100)
        time.sleep(0.05)
        winsound.Beep(350, 100)
        
    elif alert_type == 'braking':
        # Profile: Attention Chime (Sudden Braking)
        # 2 crisp, mid-tone beeps to draw the eyes forward
        winsound.Beep(1200, 200)
        time.sleep(0.1)
        winsound.Beep(1200, 200)

    elif alert_type == 'fcw':
        # Profile: Collision Imminent Panic Alarm
        # 5 extremely rapid, high-pitched piercing shrieks
        for _ in range(5):
            winsound.Beep(2500, 80)
            time.sleep(0.05)
# ==========================================
# LANE DETECTION PIPELINE
# ==========================================
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

def draw_lane_lines(img, left_line, right_line, color=[0, 255, 0]):
    line_img = np.zeros_like(img)
    poly_pts = np.array([[
        (left_line[0], left_line[1]),
        (left_line[2], left_line[3]),
        (right_line[2], right_line[3]),
        (right_line[0], right_line[1])
    ]], dtype=np.int32)
    cv2.fillPoly(line_img, poly_pts, color)
    return cv2.addWeighted(img, 0.6, line_img, 0.4, 0.0)

def pipeline(image, vehicle_boxes):
    height, width = image.shape[:2]

    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]

    color_filtered_image = select_white_yellow(image)
    
    for box in vehicle_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(color_filtered_image, (max(0, x1-10), max(0, y1-10)), (min(width, x2+10), min(height, y2+10)), (0, 0, 0), -1)

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

    # Mathematical formulas for the lines
    poly_left = None
    poly_right = None
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

    # Returning the mathematical functions so we can calculate depth perspective
    return lane_image, poly_left, poly_right, left_x_start, right_x_start

def estimate_distance(bbox_width):
    focal_length = 800  
    known_width = 1.8   
    if bbox_width == 0: return 999
    return (known_width * focal_length) / bbox_width

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
def process_video(source=0):
    global last_beep_time
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'yolov8m.pt') # Using Medium model
    model = YOLO(model_path) 
    
    if torch.cuda.is_available():
        print("CUDA detected! Offloading YOLO to GPU...")
        model.to('cuda')

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    # Setup Fullscreen
    cv2.namedWindow('ADAS Project - Live Feed', cv2.WINDOW_NORMAL)
    ##cv2.setWindowProperty('ADAS Project - Live Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    cv2.namedWindow('ADAS Project - Live Feed', cv2.WINDOW_NORMAL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    prev_closest_distance = 999
    prev_time = time.time()
    
    # State variables for Lane Smoothing
    smooth_bottom_left = None
    smooth_bottom_right = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        resized_frame = cv2.resize(frame, (1920, 1080))
        frame_center_x = 1920 // 2

        results = model(resized_frame, classes=[0, 1, 2, 3, 5, 7], conf=0.40, half=True, verbose=False)
        
        vehicle_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_boxes.append((x1, y1, x2, y2))

        lane_frame, poly_left, poly_right, bottom_left_x, bottom_right_x = pipeline(resized_frame, vehicle_boxes)

        # ---------------------------------------------------------
        # LANE DEPARTURE WARNING (Smoothed)
        # ---------------------------------------------------------
        # Apply Exponential Moving Average to stop micro-jitters
        if bottom_left_x != 0 and bottom_right_x != 1920:
            if smooth_bottom_left is None:
                smooth_bottom_left = bottom_left_x
                smooth_bottom_right = bottom_right_x
            else:
                smooth_bottom_left = 0.2 * bottom_left_x + 0.8 * smooth_bottom_left
                smooth_bottom_right = 0.2 * bottom_right_x + 0.8 * smooth_bottom_right
        
        drift_offset = 0
        if smooth_bottom_left is not None and smooth_bottom_right is not None:
            lane_center_bottom = (smooth_bottom_left + smooth_bottom_right) / 2
            drift_offset = frame_center_x - lane_center_bottom
            
            # Threshold bumped to 60px to allow normal lane weaving safely
            # Tuned for 1080p Webcam stability
            if abs(drift_offset) > 135: 
                cv2.putText(lane_frame, "WARNING: LANE DEPARTURE!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                
                if time.time() - last_beep_time > 1.5:
                    threading.Thread(target=play_alert, args=('ldw',), daemon=True).start()
                    last_beep_time = time.time()

        # ---------------------------------------------------------
        # FORWARD COLLISION & SUDDEN BRAKING (Perspective Fixed)
        # ---------------------------------------------------------
        collision_alert = False
        alert_text = ""
        current_closest_distance = 999
        closest_box = None

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f'{model.names[cls]} {conf:.2f}'

                bbox_width = x2 - x1
                distance = estimate_distance(bbox_width)
                object_center_x = (x1 + x2) / 2

                # PERSPECTIVE FIX: Calculate lane boundaries at the EXACT depth of the car's tires (y2)
                in_my_lane = False
                if poly_left is not None and poly_right is not None:
                    lane_width_at_car = poly_right(y2) - poly_left(y2)
                    # Sanity check: Ensure lines haven't crossed each other
                    if lane_width_at_car > 0: 
                        in_my_lane = poly_left(y2) < object_center_x < poly_right(y2)

                # Draw standard boxes (Green = Safe/Other Lane, Yellow = In our lane but safe distance)
                box_color = (0, 255, 255) if in_my_lane else (0, 255, 0)
                cv2.rectangle(lane_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(lane_frame, f"{label} {distance:.1f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                if in_my_lane and distance < current_closest_distance:
                    current_closest_distance = distance
                    closest_box = (x1, y1, x2, y2)

        if current_closest_distance < 999:
            distance_closed = prev_closest_distance - current_closest_distance
            
            if distance_closed > 0.5 and current_closest_distance < 50.0:
                collision_alert = True
                alert_type_trigger = 'braking' # Set the specific sound
                alert_text = "SUDDEN BRAKING OCCURED"
                
            elif current_closest_distance < 20.0:
                if distance_closed >= -0.3: 
                    collision_alert = True
                    alert_type_trigger = 'fcw' # Set the specific sound
                    alert_text = "ALERT:OBJECT TOO CLOSE, COLLISION ALERT!!"
            
            if collision_alert and closest_box is not None:
                cx1, cy1, cx2, cy2 = closest_box
                cv2.rectangle(lane_frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 4)
                cv2.putText(lane_frame, alert_text, (300, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                
                if time.time() - last_beep_time > 1.0:
                    # Pass the dynamic alert_type_trigger instead of a hardcoded string
                    threading.Thread(target=play_alert, args=(alert_type_trigger,), daemon=True).start()
                    last_beep_time = time.time()

        prev_closest_distance = current_closest_distance

        cv2.putText(lane_frame, f"Lane Drift: {drift_offset:.0f}px", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(lane_frame, f"FPS: {int(fps)}", (1100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('ADAS Project - Live Feed', lane_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video(source=0)