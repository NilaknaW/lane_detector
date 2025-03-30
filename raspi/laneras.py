import cv2
import numpy as np
import math
import time
from collections import deque
import subprocess

# Camera parameters for real-world conversion (adjust these based on your setup)
CAMERA_FOV = 60  # Field of view in degrees
IMAGE_WIDTH = 1080  # Image width in pixels
IMAGE_HEIGHT = 720  # Image height in pixels
GRID_SIZE_CM = 30  # The grid size in real life (cm)

# Moving average buffer size
BUFFER_SIZE = 50  # Adjust for smoother or more responsive output
x_mid_buffer = deque(maxlen=BUFFER_SIZE)
x_low_buffer = deque(maxlen=BUFFER_SIZE)

def capture_frame():
    # Use libcamera to capture a frame
    subprocess.run(["libcamera-still", "-o", "/tmp/frame.jpg", "-n", "--width", str(IMAGE_WIDTH), "--height", str(IMAGE_HEIGHT), "--immediate"])
    frame = cv2.imread("/tmp/frame.jpg")
    return frame

def detect_grid(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect grid lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    vertical_lines = []  # List to store only vertical lines
    horizontal_lines = []  # List to store only horizontal lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate the angle of the line with respect to the horizontal axis
            delta_x = x2 - x1
            delta_y = y2 - y1
            angle = math.atan2(delta_y, delta_x)
            
            # Consider lines with angles close to 90° (i.e., vertical lines)
            if abs(angle) > np.pi / 4:  # angle near 90 degrees or -90 degrees
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                vertical_lines.append((x1, y1, x2, y2))
            
            else:
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Draw non-vertical lines in blue
                horizontal_lines.append((x1, y1, x2, y2))

    return frame, vertical_lines, horizontal_lines

def intersect_point_x(line, y):
    a1, b1, a2, b2 = line
    x = a1 + (y - b1) * (a2 - a1) / (b2 - b1)
    return x

def calculate_xmid_xlow(vertical_lines, IMAGE_WIDTH, IMAGE_HEIGHT):
    x_mid_list = []
    x_low_list = []

    for line in vertical_lines:
        x_mid_temp = intersect_point_x(line, IMAGE_HEIGHT/2)
        x_mid_list.append(x_mid_temp)
        x_low_temp = intersect_point_x(line, IMAGE_HEIGHT/3*2)
        x_low_list.append(x_low_temp)

    # find the distance in mid row
    if len(x_mid_list) <1:
        return None, None, None
    
    x_mid_list_sorted = sorted(x_mid_list)
    x_mid_diff = x_mid_list_sorted[-1] - x_mid_list_sorted[0]

    x_mid = np.mean(x_mid_list)
    x_low = np.mean(x_low_list)
    return x_mid, x_low, x_mid_diff

def calculate_dev_angle(x_mid, x_low, x_mid_diff, IMAGE_WIDTH, IMAGE_HEIGHT):
    delta_x = x_mid - x_low
    delta_y = IMAGE_HEIGHT/6 # IMAGE_HEIGHT/3*2 - IMAGE_HEIGHT/2 # 1/3 of the image height
    theta_rad = np.arctan(delta_x/delta_y)
    theta_deg = np.degrees(theta_rad)

    # Calculate the deviation of the center from the middle of the image
    dev_px = x_mid - IMAGE_WIDTH / 2 - (IMAGE_HEIGHT / 2 * np.tan(theta_rad))
    
    return dev_px, theta_rad

def moving_average(buffer, new_value):
    """Update buffer and return smoothed value."""
    buffer.append(new_value)
    return sum(buffer) / len(buffer)

def main():
    print("Starting libcamera-based vision system...")
    # cap = cv2.VideoCapture(0)
    while True:
        # ret, frame = cap.read()
        frame = capture_frame()

        if frame is None:
            print("Failed to capture frame")
            continue

        # IMAGE_HEIGHT, IMAGE_WIDTH, _ = frame.shape 
        # if not ret:
        #     break
        
        processed_frame, vertical_lines, horizontal_lines = detect_grid(frame)
        print(f"Number of vertical lines: {len(vertical_lines)}")
        print(f"Number of horizontal lines: {len(horizontal_lines)}")
        
        if vertical_lines is None:
            break

        # print lane lines
        # for line in vertical_lines: 
        #     print(line)

        # calculate deviation and angle
        x_mid, x_low, x_mid_diff = calculate_xmid_xlow(vertical_lines, IMAGE_WIDTH, IMAGE_HEIGHT)
        
        # Apply moving average smoothing
        x_mid_smooth = moving_average(x_mid_buffer, x_mid)
        x_low_smooth = moving_average(x_low_buffer, x_low)
        x_mid_diff_smooth = moving_average(x_mid_buffer, x_mid_diff)
        
        # mark the mid and low points
        cv2.circle(processed_frame, (int(x_mid_smooth), IMAGE_HEIGHT//2), 5, (0, 0, 255), -1) # red
        cv2.circle(processed_frame, (int(x_low_smooth), IMAGE_HEIGHT//3*2), 5, (0, 255, 255), -1) # yellow

        dev_px, angle_rad = calculate_dev_angle(x_mid_smooth, x_low_smooth, x_mid_diff, IMAGE_WIDTH, IMAGE_HEIGHT)


        # draw the deviation line using angle_deg and low point
        x1 = int(x_low_smooth)
        y1 = IMAGE_HEIGHT//3*2
        x2 = int(x_low_smooth + 100 * np.tan(angle_rad))
        y2 = IMAGE_HEIGHT//3*2 - 100
        cv2.line(processed_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # draw the deviation line
        a1 = int(x_mid_smooth)
        b1 = IMAGE_HEIGHT//2
        a2 = int(x_mid_smooth + dev_px)
        b2 = IMAGE_HEIGHT//2
        cv2.line(processed_frame, (a1, b1), (a2, b2), (255, 0, 255), 2)

        dev_cm = dev_px * (GRID_SIZE_CM / x_mid_diff)
        angle_deg = np.degrees(angle_rad)

        # lets get a moving average of the dev_cm and angle deg for smooth output
        

        print(f"Deviation: {dev_cm:.2f} cm, Angle: {angle_deg:.2f} degrees")

        cv2.imshow('Grid Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # time.sleep(2)

    # cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


