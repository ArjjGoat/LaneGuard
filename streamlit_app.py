import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tempfile
import av

# Import the lane detection functions from your original code
from laneDeparture import (
    grayscale, gaussian_blur, canny_edge_detection, region_of_interest,
    display_lines, average_slope_intercept, make_coordinates, lane_detection
)

<<<<<<< HEAD
# Custom CSS remains the same
custom_css = """
<style>
    .stApp {
        background-color: #ffffff; /* Light background color */
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #1E3A8A; /* Darker button color for contrast */
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #3B82F6; /* Lighter hover color */
    }
    .stRadio > label {
        background-color: #E5E7EB;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-right: 0.5rem;
    }
    .stSidebar .sidebar-content {
        background-color: #F3F4F6; /* Light sidebar background */
    }
    .disclaimer {
        background-color: #FEF3C7;
        border: 1px solid #F59E0B;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
"""

class LaneDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        try:
            # Process the frame using your lane detection function
            averaged_lines = lane_detection(img)
            black_lines = display_lines(img, averaged_lines)
            lanes = cv2.addWeighted(img, 0.8, black_lines, 1, 1)
        except Exception as e:
            # If an error occurs, overlay the custom message on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "Road not detected, please rerun when you're on the road", 
                        (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            lanes = img
        
        # Add disclaimer text for the first 100 frames (about 3 seconds at 30 fps)
        if self.frame_count <= 100:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(lanes, "Road isn't detected, please direct camera towards the road!", 
                        (10, lanes.shape[0] - 20), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        return av.VideoFrame.from_ndarray(lanes, format="bgr24")

def process_video_frame(frame, frame_count):
    try:
        averaged_lines = lane_detection(frame)
        black_lines = display_lines(frame, averaged_lines)
        lanes = cv2.addWeighted(frame, 0.8, black_lines, 1, 1)
    except Exception as e:
        # If an error occurs, overlay the custom message on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Road not detected, please rerun when you're on the road", 
                    (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        lanes = frame
    
    # Add disclaimer text for the first 100 frames
    if frame_count <= 100:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(lanes, "Road isn't detected, please direct camera towards the road!", 
                    (10, lanes.shape[0] - 20), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
    return lanes

=======
class LaneDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process the frame using your lane detection function
        averaged_lines = lane_detection(img)
        black_lines = display_lines(img, averaged_lines)
        lanes = cv2.addWeighted(img, 0.8, black_lines, 1, 1)
        
        return av.VideoFrame.from_ndarray(lanes, format="bgr24")

>>>>>>> parent of f3fec45 (asdf)
def main():
    st.set_page_config(page_title="Lane Detection App", layout="wide")
    
    st.title("Lane Detection App")
    
<<<<<<< HEAD
    # Add a big button at the top to lead users to the demo
    if st.button("Aren't on the road? Check out a demo", 
                 key="demo_button", 
                 help="Click to see a pre-recorded demo of lane detection"):
        st.session_state.app_mode = "Laneguard Demo"
        st.rerun()
    
=======
>>>>>>> parent of f3fec45 (asdf)
    st.sidebar.title("Settings")
    input_type = st.sidebar.radio("Select input type:", ("Live Video", "Upload MP4"))
    
    if input_type == "Live Video":
        st.write("Live video processing:")
        st.markdown('<div class="disclaimer">Road isn\'t detected, please direct camera towards the road!</div>', unsafe_allow_html=True)
        webrtc_ctx = webrtc_streamer(
            key="lane-detection",
            video_transformer_factory=LaneDetectionTransformer,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )
    
    else:
        st.write("Upload an MP4 file for lane detection:")
        uploaded_file = st.file_uploader("Choose an MP4 file", type="mp4")
        
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            st.video(tfile.name)
            
            if st.button("Process Video"):
                st.markdown('<div class="disclaimer">Road isn\'t detected, please direct camera towards the road!</div>', unsafe_allow_html=True)
                cap = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
<<<<<<< HEAD
                    frame_count += 1
                    processed_frame = process_video_frame(frame, frame_count)
=======
                    averaged_lines = lane_detection(frame)
                    black_lines = display_lines(frame, averaged_lines)
                    lanes = cv2.addWeighted(frame, 0.8, black_lines, 1, 1)
>>>>>>> parent of f3fec45 (asdf)
                    
                    lanes_rgb = cv2.cvtColor(lanes, cv2.COLOR_BGR2RGB)
                    stframe.image(lanes_rgb, channels="RGB", use_column_width=True)
                
                cap.release()
    
    st.markdown("""
    ## How to use:
    1. Select input type (Live Video or Upload MP4) from the sidebar.
    2. For live video, click "Start" and allow camera access.
    3. For MP4, upload a video file and click "Process Video".
    4. Watch the processed video with lane detection!
    """)
    
    st.markdown("""
    ## About Lane Detection
    This app uses computer vision techniques to detect lanes in videos.
    It processes each frame to identify and highlight lane markings.
    """)

if __name__ == "__main__":
    main()