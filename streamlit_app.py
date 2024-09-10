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

class LaneDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process the frame using your lane detection function
        averaged_lines = lane_detection(img)
        black_lines = display_lines(img, averaged_lines)
        lanes = cv2.addWeighted(img, 0.8, black_lines, 1, 1)
        
        return av.VideoFrame.from_ndarray(lanes, format="bgr24")

def main():
    st.set_page_config(page_title="Lane Detection App", layout="wide")
    
    st.title("Lane Detection App")
    
    st.sidebar.title("Settings")
    input_type = st.sidebar.radio("Select input type:", ("Live Video", "Upload MP4"))
    
    if input_type == "Live Video":
        st.write("Live video processing:")
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
                cap = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    averaged_lines = lane_detection(frame)
                    black_lines = display_lines(frame, averaged_lines)
                    lanes = cv2.addWeighted(frame, 0.8, black_lines, 1, 1)
                    
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