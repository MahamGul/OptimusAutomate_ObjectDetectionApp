import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

# Load model (cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("Object Detection App (YOLOv8)")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    # Convert uploaded file to image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image_np, caption="Uploaded Image", use_container_width=True)

    # Run YOLO detection
    results = model(image_np)

    # Plot results (YOLO already draws boxes)
    annotated_frame = results[0].plot()

    st.subheader("Detected Image")
    st.image(annotated_frame, use_container_width=True)

    # Show detections in text form
    st.subheader("Detections")

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        st.write(f"{label} - {conf:.2f}")