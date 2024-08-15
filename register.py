import os
import random
import joblib
import shutil
import cv2 as cv
import numpy as np
import streamlit as st
from uuid import uuid4

from utils import get_embeddings_dataset, finetune
from functions import resize_image
from loader import (
    get_yolov8m_mask, 
    get_yolov8n_face, 
    get_facenet_512, 
    get_classifier, 
    get_encoder, 
)


st.header("Register Face🙋")

with st.spinner("Preparing all AI to be ready..."):
    detector_mask = get_yolov8m_mask()
    detector_face = get_yolov8n_face()
    verificator_face = get_facenet_512()
    
    encoder = get_encoder()
    classifier = get_classifier()

path_file = None
path_file_annotated = None
frames_view = []
frames_save = []
uploaded_file = None
val_size = .2

# --- retrain form
form = st.form(key='form_retrain')

name = form.text_input(label='Your name here:')
uploaded_file = form.file_uploader("**&mdash; Drop a video file of your face in 360° and give your wonderful smile. (AVI/MP4/MOV/MKV)**", accept_multiple_files=False, type=['avi', 'mp4', 'mov', 'mkv'])

submit_button = form.form_submit_button(label='Submit')

if submit_button:
    name = name.upper()
    
    path_dataset_train = os.path.join('static', 'dataset', 'train', name)
    path_dataset_val = os.path.join('static', 'dataset', 'val', name)
    path_embedding = os.path.join('static', 'embedding')
    path_embedding_train = os.path.join('static', 'embedding', 'train.npz')
    path_embedding_val = os.path.join('static', 'embedding', 'val.npz')
    
    if not name:
        st.error('Name is required!', icon='🚨')
        st.stop()
    
    if os.path.exists(path_dataset_train) or os.path.exists(path_dataset_val):
        if os.path.exists(path_dataset_train) and os.path.exists(path_dataset_val):
            st.error('User already registered.', icon='🚨')
            st.stop()
        else:
            if os.path.exists(path_dataset_train):
                shutil.rmtree(path_dataset_train)
            if os.path.exists(path_dataset_val):
                shutil.rmtree(path_dataset_val)
            
            st.info("User files are redundant. But, we fix it for now. Don't worry.", icon='ℹ️')
    
    if not uploaded_file:
        st.error('Video is required!', icon='🚨')
        st.stop()
    
    os.makedirs(path_dataset_train, exist_ok=True)
    os.makedirs(path_dataset_val, exist_ok=True)
    
    # --- save uploaded file
    _, type_file = uploaded_file.name, uploaded_file.type.split('/')[-1]

    name_file = f"{str(uuid4().hex)}.{type_file}"
    path_file = os.path.join('static', 'registration', name_file)
    
    with open(path_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # --- ensure that uploaded file is exist.
    if not os.path.exists(path_file):
        st.error('Unexpected error, saved file is missing!', icon='🚨')
        st.stop()
    else:
        st.success("File saved successfully.")

    # --- perform face detection
    bar_text = 'Analyzing video...'
    bar = st.progress(0, text=bar_text)
    
    cap = cv.VideoCapture(path_file)
    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    i = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- mask detection
        mask = detector_mask.inference(frame)
        if mask:
            if mask[0]['class_id'] == 0 or mask[0]['class_id'] == 2:
                scale = mask[0]['scale']
                x, y, w, h = mask[0]['box']
                x1, y1, x2, y2 = int(x*scale), int(y*scale), int((x + w)*scale), int((y + h)*scale)

                cv.rectangle(frame, (x1, y1), (x2, y2), [255, 0, 0], 2)
                
                frame, _, _, _, _ = resize_image(frame)
                
                frames_view.append(frame)
                bar.progress(min(i / n_frames, 1.), text=bar_text)
                
                i += 1
                
                continue
                
        # --- face detection
        face, bbox = detector_face.inference(frame)
        if face.size != 0:
            x, y, w, h = bbox
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        frame, _, _, _, _ = resize_image(frame)
        
        frames_view.append(frame)
        frames_save.append(face)
        bar.progress(min(i / n_frames, 1.), text=bar_text)
        
        i += 1
    
    cap.release()
    bar.empty()
    
    # --- if more than 50 faces data has been collected, then do random pick
    n_frames = len(frames_save)
    print('Total Frames:', len(frames_save))
    if n_frames > 50:
        ids_used = np.linspace(0, n_frames - 1, 50).astype(int)
        print('ids_used', ids_used)
        frames_save = [frames_save[idx] for idx in ids_used]
        print('Total Frames Used:', len(frames_save))
    
    # --- random split
    random.shuffle(frames_save)
    val_size = int(len(frames_save) * val_size)
    data_train, data_val = frames_save[:-val_size], frames_save[-val_size:]
    
    # --- save images of face detection results.
    for item_train in data_train:
        path_ = os.path.join(path_dataset_train, f"{str(uuid4().hex)}.jpg")
        cv.imwrite(path_, item_train)
    
    for item_val in data_val:
        path_ = os.path.join(path_dataset_val, f"{str(uuid4().hex)}.jpg")
        cv.imwrite(path_, item_val)
    
    # --- perform facenet inferences then build/update dataset.
    # --- if exist, update the existing embeddings
    x_train, y_train = get_embeddings_dataset(data_train, name)
    x_val, y_val = get_embeddings_dataset(data_val, name)
    
    if os.path.exists(path_embedding_train):
        data = np.load(path_embedding_train)
        x_train = np.concatenate([data['x_train'], x_train], axis=0)
        y_train = np.concatenate([data['y_train'], y_train], axis=0)
    
    if os.path.exists(path_embedding_val):
        data = np.load(path_embedding_val)
        x_val = np.concatenate([data['x_val'], x_val], axis=0)
        y_val = np.concatenate([data['y_val'], y_val], axis=0)
    
    np.savez_compressed(path_embedding_train, x_train=x_train, y_train=y_train)
    np.savez_compressed(path_embedding_val, x_val=x_val, y_val=y_val)
    
    # --- prepare all dataset before fine-tune (merge base dataset with updated dataset)
    data = np.load('assets/embedding/train.npz')
    x_train = np.concatenate([data['x_train'], x_train], axis=0)
    y_train = np.concatenate([data['y_train'], y_train], axis=0)
    
    data = np.load('assets/embedding/train-aug.npz')
    x_train = np.concatenate([data['x_train'], x_train], axis=0)
    y_train = np.concatenate([data['y_train'], y_train], axis=0)
    
    data = np.load('assets/embedding/val.npz')
    x_val = np.concatenate([data['x_val'], x_val], axis=0)
    y_val = np.concatenate([data['y_val'], y_val], axis=0)
    
    # --- perform fine-tune then save best estimator.
    finetune(
        src_train=(x_train, y_train),
        src_val=(x_val, y_val),
        dst_classifier='static/model/classifier.joblib',
        dst_encoder='static/model/encoder.joblib',
    )
    
    # --- show detection results with bounding box.
    st.header("Video Result🎥")
        
    container = st.container(border=True)
    st_img = container.empty()
    
    while True:
        for frame in frames_view:
            st_img.image(frame, channels="BGR", use_column_width=True, caption='Processed Video')
