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

st.sidebar.markdown("---")
st.sidebar.write("Developed by [Your Name]")

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