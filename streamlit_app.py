import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import tempfile
import av

# Import the lane detection functions from your original code
from laneDeparture import (
    grayscale, gaussian_blur, canny_edge_detection, region_of_interest,
    display_lines, average_slope_intercept, make_coordinates, lane_detection
)

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
</style>
"""
class LaneDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
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
        
        return av.VideoFrame.from_ndarray(lanes, format="bgr24")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
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
        
        return av.VideoFrame.from_ndarray(lanes, format="bgr24")

def process_video_frame(frame):
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
    return lanes

def main():
    st.set_page_config(page_title="Lane Detection App", layout="wide")
    
    # Apply custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose the app mode",
        ["Lane Detection", "Laneguard Demo"])
    
    if app_mode == "Lane Detection":
        lane_detection_page()
    elif app_mode == "Laneguard Demo":
        laneguard_demo_page()

def lane_detection_page():
    st.title("Lane Detection App")
    
    # Add a big red button at the top to lead users to the demo
    if st.button("Aren't on the road? Check out a demo", 
                 key="demo_button", 
                 help="Click to see a pre-recorded demo of lane detection"):
        st.session_state.app_mode = "Laneguard Demo"
        st.rerun()
    
    st.sidebar.title("Settings")
    input_type = st.sidebar.radio("Select input type:", ("Live Video", "Upload MP4"))
    
    if input_type == "Live Video":
        st.write("Live video processing:")
        webrtc_ctx = webrtc_streamer(
            key="lane-detection",
            video_transformer_factory=LaneDetectionTransformer,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    else:
        st.write("Upload an MP4 file for lane detection:")
        uploaded_file = st.file_uploader("Choose an MP4 file", type="mp4")
        
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            st.video(tfile.name)
            
            if st.button("Process Video"):
                cap = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    processed_frame = process_video_frame(frame)
                    
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    stframe.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                
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

def laneguard_demo_page():
    st.title("Laneguard Demo")
    st.write("This demo shows the output of our lane detection algorithm on a pre-processed video.")

    video_path = 'output_video_that_streamlit_can_play.mp4'
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)

    st.markdown("""
    ## About Laneguard
    Laneguard is our advanced lane detection system that helps drivers stay safely within their lanes.
    This demo showcases the system's ability to detect and highlight lane markings in real-time.
    """)

    if st.button("Back to Lane Detection", key="back_button"):
        st.session_state.app_mode = "Lane Detection"
        st.rerun()

if __name__ == "__main__":
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Lane Detection"
    
    main()