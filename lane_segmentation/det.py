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
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def select_white_yellow(image):
    # 1. Convert to HSL color space (better for detecting colors in shadow)
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    # 2. White Color Mask
    # High lightness (L) is key for white
    lower_white = np.uint8([0, 200, 0])
    upper_white = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower_white, upper_white)
    
    # 3. Yellow Color Mask
    # Hue (H) around 10-40 is yellow.
    lower_yellow = np.uint8([10, 0, 100])
    upper_yellow = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower_yellow, upper_yellow)
    
    # 4. Combine masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # 5. Apply mask to original image
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


# Function to draw the filled polygon between the lane lines
def draw_lane_lines(img, left_line, right_line, color=[0, 255, 0], thickness=10):
    line_img = np.zeros_like(img)
    poly_pts = np.array([[
        (left_line[0], left_line[1]),
        (left_line[2], left_line[3]),
        (right_line[2], right_line[3]),
        (right_line[0], right_line[1])
    ]], dtype=np.int32)
    
    # Fill the polygon between the lines
    cv2.fillPoly(line_img, poly_pts, color)
    
    # Overlay the polygon onto the original image
    img = cv2.addWeighted(img, 0.8, line_img, 0.5, 0.0)
    return img

# The lane detection pipeline
def pipeline(image):
    height = image.shape[0]
    width = image.shape[1]

    """
    region_of_interest_vertices = [
        (width * 0.1, height * 0.9),     # Bottom Left (cut 10% from side, 10% from bottom)
        (width * 0.45, height * 0.6),    # Top Left (towards center)
        (width * 0.55, height * 0.6),    # Top Right (towards center)
        (width * 0.9, height * 0.9),     # Bottom Right (cut 10% from side, 10% from bottom)
    ]

    """
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]
    #"""

    # Apply the color filter FIRST. Now the grass and dashboard are black.
    color_filtered_image = select_white_yellow(image)
    
    # Convert to grayscale (now much cleaner)
    gray_image = cv2.cvtColor(color_filtered_image, cv2.COLOR_RGB2GRAY)
    
    # Canny Edge Detection
    # We can lower the threshold now because noise is gone
    cannyed_image = cv2.Canny(gray_image, 50, 150)

    # Mask out the region of interest
    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32)
    )

    # Hough Lines (Use the "Indian Road" settings I gave you earlier)
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi/180,
        threshold=50, 
        lines=np.array([]),
        minLineLength=20,
        maxLineGap=100
    )
    """
    # Convert to grayscale and apply Canny edge detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)

    # Mask out the region of interest
    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32)
    )

    # Perform Hough Line Transformation to detect lines
    '''
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
    '''

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=2,              # Was 6. We need finer resolution (2 pixels) to trace curved/irregular roads.
        theta=np.pi/180,    # Was pi/60. We need standard 1-degree precision for better accuracy.
        threshold=50,       # Was 160. DRASTIC REDUCTION. Necessary to detect faded/worn paint.
        lines=np.array([]),
        minLineLength=20,   # Was 40. Lower this to catch short, eroded dash segments.
        maxLineGap=100      # Was 25. HUGE INCREASE. This bridges the large gaps where paint is missing.
    )
    """
    # Separating left and right lines based on slope
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    if lines is None:
        return image

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if math.fabs(slope) < 0.5:  # Ignore nearly horizontal lines
                continue
            if slope <= 0:  # Left lane
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:  # Right lane
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    # Fit a linear polynomial to the left and right lines
    min_y = int(image.shape[0] * (3 / 5))  # Slightly below the middle of the image
    max_y = image.shape[0]  # Bottom of the image

    if left_line_x and left_line_y:
        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
    else:
        left_x_start, left_x_end = 0, 0  # Defaults if no lines detected

    if right_line_x and right_line_y:
        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
    else:
        right_x_start, right_x_end = 0, 0  # Defaults if no lines detected

    # Create the filled polygon between the left and right lane lines
    lane_image = draw_lane_lines(
        image,
        [left_x_start, max_y, left_x_end, min_y],
        [right_x_start, max_y, right_x_end, min_y]
    )

    return lane_image

# Function to estimate distance based on bounding box size
def estimate_distance(bbox_width, bbox_height):
    # For simplicity, assume the distance is inversely proportional to the box size
    # This is a basic estimation, you may use camera calibration for more accuracy
    focal_length = 1000  # Example focal length, modify based on camera setup
    known_width = 2.0  # Approximate width of the car (in meters)
    distance = (known_width * focal_length) / bbox_width  # Basic distance estimation
    return distance

# Main function to read and process video with YOLOv8
def process_video(file='./video/car.mp4'):
    # Load the YOLOv8 model
    model = YOLO('./weights/yolo26x.pt')
    model.to('cuda')

    # Open the video file
    cap = cv2.VideoCapture(file)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # --- INSERTION 1: INITIALIZE WRITER ---
    # We use 'mp4v' codec for .mp4 files.
    # CRITICAL: Resolution (1280, 720) MUST match your cv2.resize() below.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    output_filename = 'output.mp4'
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (1280, 720))
    print(f"Recording mission to: {output_filename}")
    # --------------------------------------

    # Set the desired frame rate
    target_fps = 30
    frame_time = 1.0 / target_fps  # Time per frame to maintain 30fps

    # Resize to 720p (1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Loop through each frame
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Resize frame to 720p
        resized_frame = cv2.resize(frame, (1280, 720))

        # Run the lane detection pipeline
        lane_frame = pipeline(resized_frame)

        # Run YOLOv8 to detect cars in the current frame
        # results = model(resized_frame)
        results = model(resized_frame, classes=[0, 1, 2, 3, 5, 7, 9, 11, 12], conf=0.55, device=0)

        # Process the detections from YOLOv
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class ID

                # Only draw bounding boxes for cars with confidence >= 0.5
                #if model.names[cls] == 'car' and conf >= 0.5:
                label = f'{model.names[cls]} {conf:.2f}'

                    # Draw the bounding box
                cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(lane_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # Estimate the distance of the car
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                distance = estimate_distance(bbox_width, bbox_height)

                    # Display the estimated distance
                distance_label = f'Distance: {distance:.2f}m'
                cv2.putText(lane_frame, distance_label, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the resulting frame with both lane detection and car detection
        cv2.imshow('Lane and Car Detection', lane_frame)

        # --- TACTICAL INSERTION: WATERMARK ---
        watermark_text = "Major project"
        
        # Calculate position: Bottom Right corner
        # We start at width - 350 to give enough space for the text
        text_position = (lane_frame.shape[1] - 350, 40)
        
        # 1. Draw black outline (for visibility on any background)
        cv2.putText(
            lane_frame, 
            watermark_text, 
            text_position, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8,            # Font Scale
            (0, 0, 0),      # Color: Black (Outline)
            4,              # Thickness: Thicker than main text
            cv2.LINE_AA
        )
        
        # 2. Draw white main text over it
        cv2.putText(
            lane_frame, 
            watermark_text, 
            text_position, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8,            # Font Scale
            (255, 255, 255),# Color: White
            2,              # Thickness: Normal
            cv2.LINE_AA
        )
        # -------------------------------------
        out.write(lane_frame)
        # Limit the frame rate to 30fps
        time.sleep(frame_time)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the video processing function
if __name__ == "__main__":
    if len(sys.argv) == 2:
        process_video(sys.argv[1])
    else:
        process_video()
