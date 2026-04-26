import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import math
import threading
import winsound
import os

# --- GLOBAL VARIABLES FOR STABILITY ---
prev_poly_left = None
prev_poly_right = None
last_beep_time = 0

# --- AUDIO THREADING ---
def play_alert(alert_type):
    if alert_type == 'ldw':
        # Rumble strip sound
        winsound.Beep(350, 100)
        time.sleep(0.05)
        winsound.Beep(350, 100)
    elif alert_type == 'braking':
        # Sudden Braking Chime
        winsound.Beep(1200, 200)
        time.sleep(0.1)
        winsound.Beep(1200, 200)
    elif alert_type == 'fcw':
        # Collision Imminent Panic Alarm
        for _ in range(5):
            winsound.Beep(2500, 80)
            time.sleep(0.05)

def estimate_distance(bbox_width):
    # Pinhole camera model approximation (Tune 1500 for your specific webcam focal length)
    return max(1.0, (1.8 * 1500) / bbox_width)

# --- LANE DETECTION PIPELINE ---
def select_white_yellow(image):
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    # White color mask
    lower_white = np.uint8([0, 200, 0])
    upper_white = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower_white, upper_white)
    
    # Yellow color mask
    lower_yellow = np.uint8([10, 0, 100])
    upper_yellow = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower_yellow, upper_yellow)
    
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=mask)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def pipeline(image, vehicle_boxes):
    global prev_poly_left, prev_poly_right
    height, width = image.shape[:2]
    
    # 1. Color Filtering
    color_filtered_image = select_white_yellow(image)
    
    # 2. Dynamic Masking (With 20px padding to hide vehicle shadows)
    for box in vehicle_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(color_filtered_image, 
                      (max(0, x1-20), max(0, y1-20)), 
                      (min(width, x2+20), min(height, y2+20)), 
                      (0, 0, 0), thickness=-1)

    # 3. Grayscale & Canny
    gray_image = cv2.cvtColor(color_filtered_image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 50, 150)
    
    # 4. Region of Interest
    region_of_interest_vertices = [
        (0, height),
        (width / 2 - 100, height / 2 + 50),
        (width / 2 + 100, height / 2 + 50),
        (width, height),
    ]
    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32)
    )
    
    # 5. Hough Transform
    lines = cv2.HoughLinesP(
        cropped_image, rho=2, theta=np.pi/180, threshold=50,
        lines=np.array([]), minLineLength=40, maxLineGap=100
    )
    
    left_line_x, left_line_y, right_line_x, right_line_y = [], [], [], []
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 - x1 == 0: continue # Prevent divide by zero
                slope = (y2 - y1) / (x2 - x1)
                
                # Strict slope filtering to ignore horizontal bumper noise
                if math.fabs(slope) < 0.5: continue
                
                # Strict side filtering
                if slope < 0 and x1 < width / 2 and x2 < width / 2:
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                elif slope > 0 and x1 > width / 2 and x2 > width / 2:
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])

    # 6. Coefficient Polyfit & EMA Smoothing
    # Fitting x as a function of y (x = Ay + B) handles near-vertical lines better
    current_poly_left, current_poly_right = None, None
    
    if left_line_x and left_line_y:
        current_poly_left = np.polyfit(left_line_y, left_line_x, 1)
    if right_line_x and right_line_y:
        current_poly_right = np.polyfit(right_line_y, right_line_x, 1)

    # Smooth Left Line
    if current_poly_left is not None:
        if prev_poly_left is not None:
            prev_poly_left = 0.2 * current_poly_left + 0.8 * prev_poly_left
        else:
            prev_poly_left = current_poly_left

    # Smooth Right Line
    if current_poly_right is not None:
        if prev_poly_right is not None:
            prev_poly_right = 0.2 * current_poly_right + 0.8 * prev_poly_right
        else:
            prev_poly_right = current_poly_right

    # 7. Draw the Polygon
    lane_image = np.zeros_like(image)
    min_y = int(height * 0.6)
    max_y = height
    
    poly_left_func = None
    poly_right_func = None

    if prev_poly_left is not None and prev_poly_right is not None:
        poly_left_func = np.poly1d(prev_poly_left)
        poly_right_func = np.poly1d(prev_poly_right)
        
        left_x_start = int(poly_left_func(max_y))
        left_x_end = int(poly_left_func(min_y))
        right_x_start = int(poly_right_func(max_y))
        right_x_end = int(poly_right_func(min_y))
        
        # Prevent polygon crossover/pinching
        if left_x_end < right_x_end:
            poly_pts = np.array([[
                (left_x_start, max_y),
                (left_x_end, min_y),
                (right_x_end, min_y),
                (right_x_start, max_y)
            ]], dtype=np.int32)
            cv2.fillPoly(lane_image, poly_pts, [0, 255, 0])

    output_image = cv2.addWeighted(image, 1.0, lane_image, 0.4, 0.0)
    
    return output_image, poly_left_func, poly_right_func

# --- MAIN EXECUTION ---
def main():
    global last_beep_time
    
    print("Loading YOLOv8m...")
    model = YOLO("yolov8m.pt")
    
    if torch.cuda.is_available():
        print("CUDA detected! Offloading YOLO to GPU...")
        model.to('cuda')
        
    cap = cv2.VideoCapture(0) # Change to your video file path for testing
    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    cv2.namedWindow('ADAS Project - Live Feed', cv2.WINDOW_NORMAL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    prev_time = time.time()
    prev_closest_distance = 999
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        #frame = cv2.resize(frame, (1920, 1080))
        height, width = frame.shape[:2]
        frame_center_x = width // 2
        
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # --- Deep Learning Inference ---
        results = model(frame, classes=[0, 1, 2, 3, 5, 7], conf=0.40, half=True, verbose=False)
        
        vehicle_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_boxes.append((x1, y1, x2, y2))
                
        # --- Lane Processing ---
        lane_frame, poly_left, poly_right = pipeline(frame, vehicle_boxes)
        
        # --- LDW Engine ---
        drift_offset = 0
        if poly_left is not None and poly_right is not None:
            lane_center_bottom = (poly_left(height) + poly_right(height)) / 2
            drift_offset = frame_center_x - lane_center_bottom
            
            if abs(drift_offset) > 135:
                cv2.putText(lane_frame, "WARNING: LANE DEPARTURE!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                if time.time() - last_beep_time > 1.5:
                    threading.Thread(target=play_alert, args=('ldw',), daemon=True).start()
                    last_beep_time = time.time()

        # --- FCW Engine ---
        current_closest_distance = 999
        closest_box = None
        collision_alert = False
        alert_text = ""
        alert_type_trigger = ""

        # Draw generic boxes and find closest vehicle
        for i, box in enumerate(vehicle_boxes):
            x1, y1, x2, y2 = box
            bbox_width = x2 - x1
            distance = estimate_distance(bbox_width)
            object_center_x = (x1 + x2) / 2
            
            in_my_lane = False
            if poly_left is not None and poly_right is not None:
                lane_width_at_car = poly_right(y2) - poly_left(y2)
                if lane_width_at_car > 0:
                    in_my_lane = poly_left(y2) < object_center_x < poly_right(y2)
            
            box_color = (0, 255, 255) if in_my_lane else (0, 255, 0)
            cv2.rectangle(lane_frame, (x1, y1), (x2, y2), box_color, 2)
            
            cls_id = int(results[0].boxes[i].cls[0])
            conf = results[0].boxes[i].conf[0]
            label = f'{model.names[cls_id]} {conf:.2f} {distance:.1f}m'
            cv2.putText(lane_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            if in_my_lane and distance < current_closest_distance:
                current_closest_distance = distance
                closest_box = (x1, y1, x2, y2)

        # Threat Assessment
        if current_closest_distance < 999:
            distance_closed = prev_closest_distance - current_closest_distance
            
            if distance_closed > 0.5 and current_closest_distance < 50.0:
                collision_alert = True
                alert_type_trigger = 'braking'
                alert_text = "SUDDEN BRAKING OCCURED"
            elif current_closest_distance < 20.0:
                if distance_closed >= -0.3:
                    collision_alert = True
                    alert_type_trigger = 'fcw'
                    alert_text = "ALERT: OBJECT TOO CLOSE, COLLISION ALERT!!"
                    
        # Render Warnings
        if collision_alert and closest_box is not None:
            cx1, cy1, cx2, cy2 = closest_box
            cv2.rectangle(lane_frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 4)
            cv2.putText(lane_frame, alert_text, (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            
            if time.time() - last_beep_time > 1.0:
                threading.Thread(target=play_alert, args=(alert_type_trigger,), daemon=True).start()
                last_beep_time = time.time()

        prev_closest_distance = current_closest_distance

        # UI Overlay
        cv2.putText(lane_frame, f"Lane Drift: {drift_offset:.0f}px", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(lane_frame, f"FPS: {int(fps)}", (1100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('ADAS Project - Live Feed', lane_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()