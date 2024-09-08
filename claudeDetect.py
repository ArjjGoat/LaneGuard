import cv2
import numpy as np
from scipy.signal import find_peaks

def preprocess_image(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for yellow and white
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([255, 30, 255], dtype=np.uint8)
    
    # Create masks for yellow and white colors
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Combine masks
    mask = cv2.bitwise_or(mask_yellow, mask_white)
    
    # Apply the mask to the original image
    masked = cv2.bitwise_and(image, image, mask=mask)
    
    return masked

def detect_edges(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection with automatic thresholding
    sigma = 0.33
    median = np.median(blurred)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edges = cv2.Canny(blurred, lower, upper)
    
    return edges

def region_of_interest(image):
    height, width = image.shape[:2]
    vertices = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def detect_lane_pixels(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    nwindows = 9
    window_height = int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty

def fit_polynomial(binary_warped):
    leftx, lefty, rightx, righty = detect_lane_pixels(binary_warped)
    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def draw_lanes(image, left_fitx, right_fitx, ploty):
    color_warp = np.zeros_like(image).astype(np.uint8)
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return result

def process_frame(frame):
    preprocessed = preprocess_image(frame)
    edges = detect_edges(preprocessed)
    roi = region_of_interest(edges)
    
    # Perspective transform (bird's eye view)
    src = np.float32([[0, 720], [1280, 720], [0, 0], [1280, 0]])
    dst = np.float32([[569, 720], [711, 720], [0, 0], [1280, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    global Minv
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(roi, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
    
    left_fitx, right_fitx, ploty = fit_polynomial(warped)
    result = draw_lanes(frame, left_fitx, right_fitx, ploty)
    
    return result

def lane_departure_warning(left_fitx, right_fitx, frame_width):
    lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
    car_center = frame_width / 2
    offset = car_center - lane_center
    
    threshold = 50  # pixels
    if abs(offset) > threshold:
        return True
    return False

def main():
    cap = cv2.VideoCapture('solidWhiteRight.mp4')  # Replace with your video file
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_frame(frame)
        
        # Lane departure warning
        left_fitx, right_fitx, _ = fit_polynomial(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY))
        if lane_departure_warning(left_fitx, right_fitx, frame.shape[1]):
            cv2.putText(processed_frame, 'Lane Departure Warning!', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Lane Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()