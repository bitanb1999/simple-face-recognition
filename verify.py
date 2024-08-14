import os
import cv2 as cv
import numpy as np
import streamlit as st

from uuid import uuid4
from time import sleep

from functions import resize_image
from loader import get_yolov8n_face, get_facenet_512, get_encoder_y_facenet, get_classifier_facenet


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    return cv.LUT(image, table)


st.header("Verify Face🔐")

with st.spinner("Preparing all AI to be ready..."):
    detector_face = get_yolov8n_face()
    
    verificator_face = get_facenet_512()
    encoder_y_facenet = get_encoder_y_facenet()
    classifier_facenet = get_classifier_facenet()

file_name = None
file_type = None
file_path = None

is_capture = st.toggle('Use Webcam')

if is_capture:
    file_uploaded = st.camera_input('Take a picture.')
    
    if file_uploaded is not None:
        file_name, file_type = str(uuid4().hex), 'jpg'
        
        file_name_new = f"{file_name}.{file_type}"
        file_path = os.path.join('static', 'attendance', file_name_new)
        
        with open(file_path, 'wb') as f:
            f.write(file_uploaded.getbuffer())
else:
    file_uploaded = st.file_uploader("Drop an image that contains your face to verify you attendance. (JPG/PNG)", accept_multiple_files=False, type=['jpg', 'png'])

    if file_uploaded is not None:
        file_name, file_type = file_uploaded.name.split('.')
        
        file_name_new = f"{str(uuid4().hex)}.{file_type}"
        file_path = os.path.join('static', 'attendance', file_name_new)

        with open(file_path, "wb") as f:
            f.write(file_uploaded.getbuffer())

if file_path:
    img = cv.imread(file_path)

    face, bbox = detector_face.inference(img)
    if face.size == 0:
        # img = cv.detailEnhance(img, sigma_s=100, sigma_r=0.55)
        # face, bbox = detector_face.inference(img)
        
        img = adjust_gamma(img, gamma=1.5)
        # img = cv.detailEnhance(img, sigma_s=100, sigma_r=0.55)
        face, bbox = detector_face.inference(img)
    
    if face.size == 0:
        st.error('Face not detected', icon="❗")
        
        container = st.container(border=True)
        container.image(img, caption=f"{file_name}.{file_type}", use_column_width='always', channels='BGR')
    else:
        embedding = verificator_face.inference(img)
        if embedding.size == 0:
            pass
        else:
            img = detector_face.draw_detections(img, bbox)
            
            embedding = embedding.reshape(1, -1)
            prediction = classifier_facenet.predict_proba(embedding)
            score = np.amax(prediction, axis=1)[0]
            label = np.argmax(prediction, axis=1)[0]
            name = encoder_y_facenet.inverse_transform([label])[0]
            
            print('Label:', label)
            print('Name:', name)
            print('Score:', score)
            
            st.header('Image Result📸')
            if score < .6:
                st.info('Unknown', icon='ℹ️')
            else:
                st.success(name, icon="✅")
            
            img, _, _, _, _ = resize_image(img)
            container = st.container(border=True)
            container.image(img, caption=f"{file_name}.{file_type}", use_column_width='always', channels='BGR')
    
