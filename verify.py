import os
import cv2 as cv
import numpy as np
import streamlit as st

from uuid import uuid4
from time import sleep

from functions import resize_image
from loader import (
    get_yolov8m_mask, 
    get_yolov8n_face, 
    get_facenet_512, 
    get_classifier, 
    get_encoder, 
)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    return cv.LUT(image, table)


st.header("Verify Face🔐")

with st.spinner("Preparing all AI to be ready..."):
    detector_mask = get_yolov8m_mask()
    detector_face = get_yolov8n_face()
    verificator_face = get_facenet_512()
    
    encoder = get_encoder()
    classifier = get_classifier()

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
    
    # --- mask detection
    mask = detector_mask.inference(img)
    if mask and (mask[0]['class_id'] == 0 or mask[0]['class_id'] == 2):
        st.info('Your photo looks like you are wearing a mask. If this is the case, please remove the mask before performing facial verification.', icon='ℹ')
        scale = mask[0]['scale']
        x, y, w, h = mask[0]['box']
        x1, y1, x2, y2 = int(x*scale), int(y*scale), int((x + w)*scale), int((y + h)*scale)

        cv.rectangle(img, (x1, y1), (x2, y2), [255, 0, 0], 2)
        
        img, _, _, _, _ = resize_image(img)
    else:
        # --- face detection
        face, bbox = detector_face.inference(img)

        if face.size == 0:
            # img = cv.detailEnhance(img, sigma_s=100, sigma_r=0.55)
            # face, bbox = detector_face.inference(img)
            
            img = adjust_gamma(img, gamma=1.5)
            # img = cv.detailEnhance(img, sigma_s=100, sigma_r=0.55)
            face, bbox = detector_face.inference(img)

        if face.size == 0:
            st.error('Your face is not detected. Make sure the lighting is sufficient and the face is clearly visible.', icon="❗")
        
        # --- face verification
        if face.size != 0:
            embedding = verificator_face.inference(face)
            if embedding.size == 0:
                st.error("Something went wrong. Your face can't be analyzed. Please try using different face.", icon="❗")
            else:
                img = detector_face.draw_detections(img, bbox)
                
                embedding = embedding.reshape(1, -1)
                prediction = classifier.predict_proba(embedding)
                for i, pred in enumerate(prediction[0]):
                    print(i, pred)
                
                score = np.amax(prediction, axis=1)[0]
                label = np.argmax(prediction, axis=1)[0]
                name = encoder.inverse_transform([label])[0]
                
                print('Label:', label)
                print('Name:', name)
                print('Score:', score)
                
                st.header('Image Result📸')
                if score < .5:
                    st.info('Unknown', icon='ℹ️')
                else:
                    st.success(name, icon="✅")
                
                img, _, _, _, _ = resize_image(img)
    
    container = st.container(border=True)
    container.image(img, caption=f"{file_name}.{file_type}", use_column_width='always', channels='BGR')
    
