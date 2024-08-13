import cv2
import numpy as np
import time
from pygame import mixer

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gaussian_blur(image, kernel_size=(3, 3)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def canny_edge_detection(image, low_threshold=150, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

def region_of_interest(image):
    height, width = image.shape
    triangle = np.array([[(50, height), (int(width/2), int(height/2)+90), (width-50, height)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    return cv2.bitwise_and(image, mask)

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image

def average_slope_intercept(image, lines):
    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = parameters

            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))

    left_avg = np.average(left_lines, axis=0) if left_lines else (1, 1)
    right_avg = np.average(right_lines, axis=0) if right_lines else (1, 1)

    return np.array([
        make_coordinates(image, left_avg),
        make_coordinates(image, right_avg)
    ])

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3.75/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def lane_detection(frame):
    gray = grayscale(frame)
    blurred = gaussian_blur(gray)

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([75, 100, 100], dtype="uint8")
    upper_yellow = np.array([255, 165, 165], dtype="uint8")

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray, 150, 175)
    mask_combined = cv2.bitwise_or(mask_white, mask_yellow)
    masked_image = cv2.bitwise_and(gray, mask_combined)

    edges = canny_edge_detection(masked_image)
    roi = region_of_interest(edges)

    lines = cv2.HoughLinesP(roi, 2, np.pi/180, 100, np.array([]), minLineLength=100, maxLineGap=100)
    averaged_lines = average_slope_intercept(frame, lines)

    return averaged_lines

def main():
    cam = cv2.VideoCapture("solidWhiteRight.mp4")
    old_avg = np.array([])
    counter = 0
    init_time = time.time_ns()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        original_frame = frame.copy()
        car_region = frame[650:, :]
        frame = frame[:650, :]

        averaged_lines = lane_detection(frame)

        if abs(averaged_lines[1][0] - averaged_lines[0][0]) <= 450:
            averaged_lines = old_avg

        old_avg = averaged_lines

        black_lines = display_lines(original_frame, averaged_lines)
        lanes = cv2.addWeighted(original_frame, 0.8, black_lines, 1, 1)

        height, width = frame.shape[:2]
        now = time.time_ns()

        if (abs(averaged_lines[1][0] - width >= 425) or abs(averaged_lines[0][0] - width >= 425)) and (now - init_time) > 300000000:
            init_time = now
            counter += 1
            print(counter)
            mixer.init()
            sound = mixer.Sound("info.wav")
            sound.play()

        lanes = cv2.arrowedLine(lanes, (width, int(height/2)), (width, int(height/2)-100), (0, 255, 0), 10)

        cv2.imshow("Lane Detection", lanes)

        if cv2.waitKey(1) != -1:
            break

    cam.release()
    cv2.destroyAllWindows()
    return counter

if __name__ == "__main__":
    main()