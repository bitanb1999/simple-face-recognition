import os
import random
import shutil
import cv2 as cv
import numpy as np
import streamlit as st

from uuid import uuid4
from loader import get_yolov8n_face, get_facenet_512, get_encoder_y_facenet, get_classifier_facenet
from functions import resize_image


st.header("Register Face🙋")

with st.spinner("Preparing all AI to be ready..."):
    detector_face = get_yolov8n_face()
    
    verificator_face = get_facenet_512()
    encoder_y_facenet = get_encoder_y_facenet()
    classifier_facenet = get_classifier_facenet()

path_file = None
path_file_annotated = None
frames_view = []
frames_save = []
uploaded_file = None
val_size = .2

# --- retrain form
form = st.form(key='form_retrain')

name = form.text_input(label='Your name here:')
uploaded_file = form.file_uploader("**&mdash; Drop a video file that contains your face in 360°. (AVI/MP4/MOV/MKV)**", accept_multiple_files=False, type=['avi', 'mp4', 'mov', 'mkv'])

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
    
    # --- save detection results, i.e. face frame.
    os.makedirs(path_dataset_train)
    os.makedirs(path_dataset_val)
    
    for item_train in data_train:
        path_ = os.path.join(path_dataset_train, f"{str(uuid4().hex)}.jpg")
        cv.imwrite(path_, item_train)
    
    for item_val in data_val:
        path_ = os.path.join(path_dataset_val, f"{str(uuid4().hex)}.jpg")
        cv.imwrite(path_, item_val)
    
    # --- save embedding of splitted data
    os.makedirs(path_embedding, exist_ok=True)
    
    x_train = []
    y_train = []
    for item_train in data_train:
        embedding = verificator_face.inference(item_train).squeeze()
        if embedding.size == 0:
            continue
        
        x_train.append(embedding)
        y_train.append(name)
    else:
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
    
    x_val = []
    y_val = []
    for item_val in data_val:
        embedding = verificator_face.inference(item_val).squeeze()
        if embedding.size == 0:
            continue
        
        x_val.append(embedding)
        y_val.append(name)
    else:
        x_val = np.asarray(x_val)
        y_val = np.asarray(y_val)
    
    if os.path.exists(path_embedding_train):
        data = np.load(path_embedding_train)
        
        x_train = np.concatenate([data['x_train'], x_train], axis=0)
        y_train = np.concatenate([data['y_train'], y_train], axis=0)
    
    np.savez_compressed(path_embedding_train, x_train=x_train, y_train=y_train)
    
    if os.path.exists(path_embedding_val):
        data = np.load(path_embedding_val)
        
        x_val = np.concatenate([data['x_val'], x_val], axis=0)
        y_val = np.concatenate([data['y_val'], y_val], axis=0)
    
    np.savez_compressed(path_embedding_val, x_val=x_val, y_val=y_val)
    
    # --- show detection results with bounding box.
    st.header("Video Result🎥")
        
    container = st.container(border=True)
    st_img = container.empty()
    
    while True:
        for frame in frames_view:
            st_img.image(frame, channels="BGR", use_column_width=True, caption='Processed Video')
