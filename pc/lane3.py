import cv2
import numpy as np
import math
import time
import subprocess

# Camera parameters for real-world conversion
CAMERA_FOV = 60  # Field of view in degrees
IMAGE_WIDTH = 640  # Image width in pixels
GRID_SIZE_CM = 30  # Grid size in real life (cm)

def capture_frame():
    # Use libcamera to capture a frame
    subprocess.run(["libcamera-still", "-o", "/tmp/frame.jpg", "-n", "--width", str(IMAGE_WIDTH), "--height", "480", "--immediate"])
    frame = cv2.imread("/tmp/frame.jpg")
    return frame

def detect_grid(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    vertical_lines = []
    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            delta_x = x2 - x1
            delta_y = y2 - y1
            angle = math.atan2(delta_y, delta_x)
            if abs(angle) > np.pi / 4:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                vertical_lines.append((x1, y1, x2, y2))
            else:
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                horizontal_lines.append((x1, y1, x2, y2))
    
    return frame, vertical_lines, horizontal_lines

def calculate_center(lane_lines):
    if len(lane_lines) == 0:
        return None, None, None
    
    x_coords = [x1 for x1, _, x2, _ in lane_lines] + [x2 for _, _, x2, _ in lane_lines]
    center_x = np.mean(x_coords)

    max_y_mean = 0
    difference_y = 0
    for x1, y1, x2, y2 in lane_lines:
        y = max(y1, y2)
        if y > max_y_mean:
            max_y_mean = y
            difference_x = abs(x1 - x2)

    return center_x, max_y_mean, difference_x

def calculate_angle(center_x, image_width):
    image_center = image_width / 2
    deviation = center_x - image_center
    angle = math.degrees(math.asin((deviation / image_center) * math.sin(math.radians(CAMERA_FOV / 2))))
    return deviation, angle

def main():
    print("Starting libcamera-based vision system...")
    
    while True:
        frame = capture_frame()

        if frame is None:
            print("Failed to capture frame")
            continue
        
        processed_frame, lane_lines, horizontal_lines = detect_grid(frame)

        center_x, nearest_y, difference = calculate_center(lane_lines)
        if center_x is not None:
            deviation, angle = calculate_angle(center_x, IMAGE_WIDTH)
        
        if nearest_y is not None and difference > 0:
            distance_y = nearest_y * (GRID_SIZE_CM / difference)
            deviation_actual = deviation * (GRID_SIZE_CM / difference) # Convert to cm
            print(f"Angle: {angle:.2f} deg, Deviation: {deviation_actual:.2f} cm, Dist_y: {deviation_actual:.2f} cm")

        cv2.imshow('Grid Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
