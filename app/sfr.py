import os
import random
import shutil
import cv2 as cv
import numpy as np
import pandas as pd
from uuid import uuid4
from flasgger import swag_from
from flask import (
    Blueprint,
    request,
    current_app,
)

from loader import (
    get_yolov8m_mask,
    get_yolov8n_face,
    get_minifasnet,
    get_facenet_512,
    get_encoder,
    get_classifier,
)
from utils import get_embeddings_dataset, finetune
from functions import resize_image


bp = Blueprint('sfr', __name__, url_prefix='/api/sfr')

@bp.route('/verify', methods=['POST'])
@swag_from('apidocs/sfr/verify.yaml')
def verify():
    if request.method == 'POST':
        response = {
            'Message': None,
            'DetectionMask': [],
            'DetectionFace': [],
            'DetectionFake': [],
            'VerificationFace': [],
        }
        if 'img' not in request.files:
            response['Message'] = 'Invalid!'
            return response, 400
        
        # --- load image
        img = cv.imdecode(np.frombuffer(request.files['img'].read(), np.uint8), cv.IMREAD_COLOR)
        
        # --- detect mask
        detector_mask = get_yolov8m_mask()
        mask = detector_mask.inference(img)
        if mask:
            is_mask = True if mask[0]['class_id'] != 1 else False
            response['DetectionMask'] = [{
                'Result': is_mask,
                'Score': mask[0]['confidence'],
            }]
            if is_mask:
                return response, 400
        
        # --- detect face
        detector_face = get_yolov8n_face()
        face, bbox, score = detector_face.inference(img)
        is_face = True if face.size != 0 else False
        response['DetectionFace'] = [{
            'Result': is_face,
            'Score': score,
        }]
        if not is_face:
            return response, 400

        # --- detect spoof
        detector_liveness = get_minifasnet()
        label, score = detector_liveness.inference(img, bbox)
        is_fake = True if label != 1 and score >=.95 else False
        response['DetectionFake'] = [{
            'Result': is_fake,
            'Score': score,
        }]
        if is_fake:
            return response, 400
        
        # --- verify face
        verificator_face = get_facenet_512()
        embedding = verificator_face.inference(face)
        embedding = embedding.reshape(1, -1)
        
        classifier = get_classifier(current_app.config['PATH_CLASSIFIER'])
        prediction = classifier.predict_proba(embedding)
        score = np.amax(prediction, axis=1)[0]
        label = np.argmax(prediction, axis=1)[0]
        
        encoder = get_encoder(current_app.config['PATH_ENCODER'])
        name = encoder.inverse_transform([label])[0]
        response['VerificationFace'] = [{
            'Name': 'Unknown' if score < .5 else name
        }]
        
        return response, 200

@bp.route('/register', methods=['POST'])
@swag_from('apidocs/sfr/register.yaml')
def register():
    if request.method == 'POST':
        response = {
            'Message': None,
        }
        
        if 'name' not in request.form:
            response['Message'] = 'Invalid!'
            return response, 400
        
        if 'video' not in request.files:
            response['Message'] = 'Invalid!'
            return response, 400
        
        name = request.form['name']
        name = name.upper()
        
        path_dataset_train = os.path.join(current_app.config['PATH_DATASET_TRAIN'], name)
        path_dataset_val = os.path.join(current_app.config['PATH_DATASET_VAL'], name)
        
        if os.path.exists(path_dataset_train) or os.path.exists(path_dataset_val):
            if os.path.exists(path_dataset_train) and os.path.exists(path_dataset_val):
                response['Message'] = 'User already registered.'
                return response, 409
            else:
                if os.path.exists(path_dataset_train):
                    shutil.rmtree(path_dataset_train)
                if os.path.exists(path_dataset_val):
                    shutil.rmtree(path_dataset_val)
        
        os.makedirs(path_dataset_train, exist_ok=True)
        os.makedirs(path_dataset_val, exist_ok=True)
        
        video = request.files['video']
        video_name = video.filename
        if not video:
            response['Message'] = 'Invalid file type!'
            return response, 415
        
        if not ('.' in video_name and video_name.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'mkv'}):
            response['Message'] = 'Invalid file extension!'
            return response, 415
        
        _, ext = video_name.rsplit('.', 1)
        path_register = os.path.join(current_app.config['PATH_REGISTRATION'], f"{str(uuid4().hex)}.{ext}")
        video.save(path_register)
        
        # --- analyze video
        detector_mask = get_yolov8m_mask()
        detector_face = get_yolov8n_face()
        
        frames_save = []
        
        cap = cv.VideoCapture(path_register)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- mask detection
            mask = detector_mask.inference(frame)
            if mask:
                if mask[0]['class_id'] == 0 or mask[0]['class_id'] == 2:
                    continue
                    
            # --- face detection
            face, bbox, score = detector_face.inference(frame)
            if face.size != 0:
                x, y, w, h = bbox
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            frame, _, _, _, _ = resize_image(frame)
            
            frames_save.append(face)
        cap.release()
        
        # --- if more than 50 faces data has been collected, then do random pick
        n_frames = len(frames_save)
        print('Total Frames:', len(frames_save))
        if n_frames > 50:
            ids_used = np.linspace(0, n_frames - 1, 50).astype(int)
            print('ids_used', ids_used)
            frames_save = [frames_save[idx] for idx in ids_used]
            print('Total Frames Used:', len(frames_save))
        
        # --- random split
        val_size = .2
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
        
        if os.path.exists(current_app.config['PATH_EMBEDDING_TRAIN']):
            data = np.load(current_app.config['PATH_EMBEDDING_TRAIN'])
            x_train = np.concatenate([data['x_train'], x_train], axis=0)
            y_train = np.concatenate([data['y_train'], y_train], axis=0)
        
        if os.path.exists(current_app.config['PATH_EMBEDDING_VAL']):
            data = np.load(current_app.config['PATH_EMBEDDING_VAL'])
            x_val = np.concatenate([data['x_val'], x_val], axis=0)
            y_val = np.concatenate([data['y_val'], y_val], axis=0)
        
        np.savez_compressed(current_app.config['PATH_EMBEDDING_TRAIN'], x_train=x_train, y_train=y_train)
        np.savez_compressed(current_app.config['PATH_EMBEDDING_VAL'], x_val=x_val, y_val=y_val)
        
        # --- prepare all dataset before fine-tune (merge base dataset with updated dataset)
        data = np.load(current_app.config['PATH_ASSETS_EMBEDDING_TRAIN'])
        x_train = np.concatenate([data['x_train'], x_train], axis=0)
        y_train = np.concatenate([data['y_train'], y_train], axis=0)
        
        data = np.load(current_app.config['PATH_ASSETS_EMBEDDING_AUG'])
        x_train = np.concatenate([data['x_train'], x_train], axis=0)
        y_train = np.concatenate([data['y_train'], y_train], axis=0)
        
        data = np.load(current_app.config['PATH_ASSETS_EMBEDDING_VAL'])
        x_val = np.concatenate([data['x_val'], x_val], axis=0)
        y_val = np.concatenate([data['y_val'], y_val], axis=0)
        
        # --- perform fine-tune then save best estimator.
        finetune(
            src_train=(x_train, y_train),
            src_val=(x_val, y_val),
            dst_classifier=current_app.config['PATH_CLASSIFIER'],
            dst_encoder=current_app.config['PATH_ENCODER'],
        )
        
        return {'Message': 'Success'}, 200

@bp.route('/stored', methods=['POST'])
@swag_from('apidocs/sfr/stored.yaml')
def stored():
    if request.method == 'POST':
        encoder = get_encoder(current_app.config['PATH_ENCODER'])
        names = encoder.classes_.tolist()
        ids = encoder.transform(names)
        
        df = pd.DataFrame({'ID': ids, 'Name': names})
        df['Train'] = df.Name.apply(
            lambda x: 
                len(os.listdir(os.path.join(current_app.config['PATH_ASSETS_DATASET'], 'train', x))) 
                if x in os.listdir(os.path.join(current_app.config['PATH_ASSETS_DATASET'], 'train')) 
                else len(os.listdir(os.path.join(current_app.config['PATH_DATASET_TRAIN'], x)))
            )
        df['Validation'] = df.Name.apply(
            lambda x: 
                len(os.listdir(os.path.join(current_app.config['PATH_ASSETS_DATASET'], 'val', x))) 
                if x in os.listdir(os.path.join(current_app.config['PATH_ASSETS_DATASET'], 'val')) 
                else len(os.listdir(os.path.join(current_app.config['PATH_DATASET_VAL'], x)))
            )
        return df.to_dict(orient='records')

@bp.route('/reset', methods=['POST'])
@swag_from('apidocs/sfr/reset.yaml')
def reset():
    list_dir = os.listdir(current_app.config['PATH_STATIC'])
    if not list_dir:
        return {'Message': 'There is no data to remove.'}
    else:
        for dir in list_dir:
            dir_remove = os.path.join('static', dir)
            if os.path.exists(dir_remove):
                shutil.rmtree(dir_remove)
                os.makedirs(dir_remove)
        else:
            return {'Message': 'Success'}
