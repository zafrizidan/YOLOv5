import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Zidan/Documents/yolov5/runs/train/yolov5s_results5/weights/best.pt', force_reload=True)

st.title('Real-Time Face Mask Detection')
st.markdown('This system can identify whether a person is <span style="font-weight: bold; color: green;">wearing a face mask</span>, <span style="font-weight: bold; color: red;">not wearing a face mask</span> or <span style="font-weight: bold; color: yellow;">wearing a face mask incorrectly</span>.', unsafe_allow_html=True)

# Function to perform inference
def run_detection(image):
    results = model(image)
    return results

# Function to perform inference on an uploaded image
def detect_uploaded_image(uploaded_image):
    image = Image.open(uploaded_image)
    results = run_detection(image)
    return image, results

# Function to perform inference on webcam feed
def detect_webcam_feed():
    video_capture = cv2.VideoCapture(0)
    while True:
        _, frame = video_capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = run_detection(frame)
        st.image(frame, channels="RGB", use_column_width=True)
        st.write(results.pandas().xyxy[0])  # Show detection results in a table

# Streamlit file uploader for image upload
uploaded_file = st.file_uploader("Choose an image..", type=["jpg", "jpeg", "png"])

# Check if image uploaded
if uploaded_file is not None:
    image, results = detect_uploaded_image(uploaded_file)
    
    # Display uploaded image and detection results
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    if st.button('Detect Objects'):

        # Display detection results in table form
        st.subheader('Detection Results:')
        detection_data = []
        for detection in results.xyxy[0]:
            class_index = int(detection[5])  # Class index
            class_name = model.names[class_index]  # Get class name from model.names
            confidence = round(float(detection[4]), 2)  # Confidence score
            bounding_box = [round(float(coord), 2) for coord in detection[:4]]  # Bounding box coordinates
            detection_data.append([class_name, confidence, bounding_box])

    if detection_data:
        st.table(detection_data)
    else:
        st.write("No objects detected.")

# Streamlit webcam access
if st.button('Use Webcam for real-time detecting'):
    st.write('Accessing Webcam...')
    detect_webcam_feed()
