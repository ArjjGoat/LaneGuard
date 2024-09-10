import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
# Import the lane detection functions from your original code
from laneDeparture import (
    grayscale, gaussian_blur, canny_edge_detection, region_of_interest,
    display_lines, average_slope_intercept, make_coordinates, lane_detection
)

def process_frame(frame):
    averaged_lines = lane_detection(frame)
    black_lines = display_lines(frame, averaged_lines)
    lanes = cv2.addWeighted(frame, 0.8, black_lines, 1, 1)
    return lanes

st.set_page_config(page_title="Lane Detection App", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Lane Detection", "Laneguard Demo"))

if page == "Lane Detection":
    st.title("Lane Detection App")
    st.sidebar.title("Settings")
    input_type = st.sidebar.radio("Select input type:", ("Live Video", "Upload MP4"))

    if input_type == "Live Video":
        st.sidebar.warning("Note: Live video may not work on mobile devices.")
        if st.sidebar.button("Start Live Video"):
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = process_frame(frame)
                stframe.image(processed_frame, channels="RGB", use_column_width=True)
            cap.release()
    else:
        uploaded_file = st.sidebar.file_uploader("Choose an MP4 file", type="mp4")
       
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = process_frame(frame)
                stframe.image(processed_frame, channels="RGB", use_column_width=True)
            cap.release()

    # Main content
    st.markdown("""
    ## How to use:
    1. Select input type (Live Video or Upload MP4) from the sidebar.
    2. For live video, click "Start Live Video" (may not work on mobile).
    3. For MP4, upload a video file using the file uploader.
    4. Watch the processed video with lane detection!
    """)
    st.markdown("""
    ## About Lane Detection
    This app uses computer vision techniques to detect lanes in videos.
    It processes each frame to identify and highlight lane markings.
    """)

elif page == "Laneguard Demo":
    st.title("Laneguard Demo")
    st.write("This demo shows the output of our lane detection algorithm on a pre-processed video.")

    # Load and display the output.mp4 video
    video_path = 'output.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error(f"Error: Could not open video file {video_path}")
    else:
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.write(f"Video FPS: {fps}")
        st.write(f"Video Resolution: {frame_width}x{frame_height}")

        # Create a placeholder for the video frames
        video_placeholder = st.empty()

        # Add a button to start/stop the video
        if st.button("Play/Pause"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display the frame
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Add a slight delay to control playback speed
                cv2.waitKey(int(1000/fps))

        cap.release()

    st.markdown("""
    ## About Laneguard
    Laneguard is our advanced lane detection system that helps drivers stay safely within their lanes.
    This demo showcases the system's ability to detect and highlight lane markings in real-time.
    """)

st.sidebar.markdown("---")
st.sidebar.write("Developed by Arjun Bakhale and Ankit Rao")