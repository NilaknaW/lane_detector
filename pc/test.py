import cv2
import numpy as np
import math
import time

# Camera parameters for real-world conversion (adjust these based on your setup)
CAMERA_FOV = 60  # Field of view in degrees
IMAGE_WIDTH = 640  # Image width in pixels
GRID_SIZE_CM = 30  # The grid size in real life (cm)

def detect_grid(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect grid lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    # if lines is not None:
    #     lane_lines = []
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         lane_lines.append((x1, y1, x2, y2))
    #     return frame, lane_lines
    # return frame, []
    vertical_lines = []  # List to store only vertical lines
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
                
    return frame, vertical_lines

def calculate_center(lane_lines):
    if len(lane_lines) == 0:
        return None, None  # No lines detected
    
    # Calculate the average x-coordinate of the lines to find the center of the lane
    x_coords = []
    for x1, y1, x2, y2 in lane_lines:
        x_coords.append(x1)
        x_coords.append(x2)
    
    center_x = np.mean(x_coords)
    return center_x

def calculate_angle(center_x, image_width):
    # Calculate the deviation of the center from the middle of the image
    image_center = image_width / 2
    deviation = center_x - image_center

    # Convert this deviation to an angle
    angle = (deviation / image_center) * (CAMERA_FOV / 2)
    return angle

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, lane_lines = detect_grid(frame)

        # Calculate the center of the lane
        center_x = calculate_center(lane_lines)
        if center_x is not None:
            # Calculate the angle to the center of the lane
            angle = calculate_angle(center_x, IMAGE_WIDTH)
            print(f"Angle to center: {angle:.2f} degrees")
            
            # Calculate position in real-world coordinates (assuming grid size is 30cm)
            # Assuming camera calibration gives us a way to convert pixels to cm
            # real_position = (center_x / IMAGE_WIDTH) * GRID_SIZE_CM
            # print(f"Position (in cm): {real_position:.2f} cm")
        
        cv2.imshow('Grid Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # time.sleep(2)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
