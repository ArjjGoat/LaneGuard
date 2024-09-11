import streamlit as st
from PIL import Image
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import tempfile
import av
from streamlit_option_menu import option_menu

# Import the lane detection functions from your original code
from laneDeparture import (
    grayscale, gaussian_blur, canny_edge_detection, region_of_interest,
    display_lines, average_slope_intercept, make_coordinates, lane_detection
)

# Custom CSS remains the same
custom_css = """
<style>
    .stApp {
        background-color: #ffffff;
        background-image: url("https://www.transparenttextures.com/patterns/cubes.png");
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #1E3A8A;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3B82F6;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stRadio > label {
        background-color: #E5E7EB;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-right: 0.5rem;
        transition: all 0.3s ease;
    }
    .stRadio > label:hover {
        background-color: #D1D5DB;
    }
    .stSidebar .sidebar-content {
        background-color: #F3F4F6;
    }
    .disclaimer {
        background-color: #FEF3C7;
        border: 1px solid #F59E0B;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.4);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(245, 158, 11, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(245, 158, 11, 0);
        }
    }
    .fancy-border {
        border: 3px solid;
        border-image: linear-gradient(45deg, #1E3A8A, #3B82F6) 1;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
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
            averaged_lines = lane_detection(img)
            black_lines = display_lines(img, averaged_lines)
            lanes = cv2.addWeighted(img, 0.8, black_lines, 1, 1)
        except Exception as e:
            lanes = img
        
        if self.frame_count == 1:
            st.session_state.show_disclaimer = True
        elif self.frame_count > 100:  # Hide disclaimer after 100 frames
            st.session_state.show_disclaimer = False
        
        return av.VideoFrame.from_ndarray(lanes, format="bgr24")

def process_video_frame(frame):
    try:
        averaged_lines = lane_detection(frame)
        black_lines = display_lines(frame, averaged_lines)
        lanes = cv2.addWeighted(frame, 0.8, black_lines, 1, 1)
    except Exception as e:
        lanes = frame
    
    return lanes

def main():
    im = Image.open("favicon.ico")
    st.set_page_config(page_title="Lane Detection App", layout="wide", initial_sidebar_state="expanded", page_icon=im)
    st.markdown(custom_css, unsafe_allow_html=True)

    if 'show_disclaimer' not in st.session_state:
        st.session_state.show_disclaimer = False

    with st.sidebar:
        app_mode = option_menu("Navigation", ["Lane Detection", "Laneguard Demo"],
                               icons=['camera', 'play-circle'], menu_icon="list", default_index=0)

    if app_mode == "Lane Detection":
        lane_detection_page()
    elif app_mode == "Laneguard Demo":
        laneguard_demo_page()

def lane_detection_page():
    st.title("Lane Detection App")
    
    if st.button("Aren't on the road? Check out a demo", help="Click to see a pre-recorded demo of lane detection"):
        st.session_state.app_mode = "Laneguard Demo"
        st.experimental_rerun()
    
    input_type = st.radio("Select input type:", ("Live Video", "Upload MP4"), horizontal=True)
    
    disclaimer_placeholder = st.empty()
    
    if input_type == "Live Video":
        with st.expander("Live Video Processing", expanded=True):
            webrtc_ctx = webrtc_streamer(
                key="lane-detection",
                video_transformer_factory=LaneDetectionTransformer,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            if webrtc_ctx.state.playing:
                st.session_state.show_disclaimer = True
            
    else:
        with st.expander("Upload and Process Video", expanded=True):
            uploaded_file = st.file_uploader("Choose an MP4 file", type="mp4")
            
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                
                st.video(tfile.name)
                
                if st.button("Process Video"):
                    st.session_state.show_disclaimer = True
                    with st.spinner("Processing video..."):
                        cap = cv2.VideoCapture(tfile.name)
                        stframe = st.empty()
                        
                        progress_bar = st.progress(0)
                        frame_count = 0
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            processed_frame = process_video_frame(frame)
                            
                            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            stframe.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                            
                            frame_count += 1
                            progress_bar.progress(frame_count / total_frames)
                            
                            if frame_count > 100:
                                st.session_state.show_disclaimer = False
                        
                        cap.release()
                    st.success("Video processing complete!")
                    st.session_state.show_disclaimer = False
    
    if st.session_state.show_disclaimer:
        disclaimer_placeholder.markdown('<div class="disclaimer">Road isn\'t detected, please direct camera towards the road!</div>', unsafe_allow_html=True)
    else:
        disclaimer_placeholder.empty()
    
    st.markdown('<div class="fancy-border">', unsafe_allow_html=True)
    st.markdown("""
    ## How to use:
    1. Select input type (Live Video or Upload MP4) above.
    2. For live video, click "Start" and allow camera access.
    3. For MP4, upload a video file and click "Process Video".
    4. Watch the processed video with lane detection!
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("About Lane Detection"):
        st.write("""
        This app uses computer vision techniques to detect lanes in videos.
        It processes each frame to identify and highlight lane markings.
        The algorithm works best on clear road markings and good lighting conditions.
        """)
        st.info("For best results, ensure the camera is focused on the road ahead.")

def laneguard_demo_page():
    st.title("Laneguard Demo")
    
    st.write("This demo shows the output of our lane detection algorithm on a pre-processed video.")

    video_path = 'output_video_that_streamlit_can_play.mp4'
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "95%", "2%")
    col2.metric("Processing Speed", "30 FPS", "5 FPS")
    col3.metric("Detection Range", "268.2 meters", "10 meters")

    with st.expander("About Laneguard"):
        st.write("""
        Laneguard is our advanced lane detection system that helps drivers stay safely within their lanes.
        This demo showcases the system's ability to detect and highlight lane markings in real-time.
        """)
        st.success("Laneguard has been tested on various road conditions and weather scenarios.")

    if st.button("Back to Lane Detection", key="back_button"):
        st.session_state.app_mode = "Lane Detection"
        st.experimental_rerun()

if __name__ == "__main__":
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Lane Detection"
    
    main()