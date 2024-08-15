import os

class Config(object):
    SECRET_KEY = 'plVdTwE0roGrrdXMBzWv9SSguvg6YDJp'
    
    PATH_BASE = os.path.dirname(__file__)
    
    PATH_ASSETS_DATASET = os.path.join(PATH_BASE, 'assets', 'dataset')
    PATH_ASSETS_EMBEDDING = os.path.join(PATH_BASE, 'assets', 'embedding')
    PATH_ASSETS_MODEL = os.path.join(PATH_BASE, 'assets', 'model')
    PATH_ASSETS_PLACEHOLDER = os.path.join(PATH_BASE, 'assets', 'placeholder')
    
    PATH_ASSETS_EMBEDDING_TRAIN = os.path.join(PATH_ASSETS_EMBEDDING, 'train.npz')
    PATH_ASSETS_EMBEDDING_AUG = os.path.join(PATH_ASSETS_EMBEDDING, 'train-aug.npz')
    PATH_ASSETS_EMBEDDING_VAL = os.path.join(PATH_ASSETS_EMBEDDING, 'val.npz')
    
    PATH_DETECTOR_MASK = os.path.join(PATH_ASSETS_MODEL, 'yolov8m-mask.onnx')
    PATH_DETECTOR_FACE = os.path.join(PATH_ASSETS_MODEL, 'yolov8n-face-hpc203.onnx')
    PATH_DETECTOR_FAKE = os.path.join(PATH_ASSETS_MODEL, '2.7_80x80_MiniFASNetV2.onnx')
    PATH_VERIFICATOR_FACE = os.path.join(PATH_ASSETS_MODEL, 'facenet512_weights.onnx')
    PATH_ENCODER = os.path.join(PATH_ASSETS_MODEL, 'encoder.joblib')
    PATH_CLASSIFIER = os.path.join(PATH_ASSETS_MODEL, 'classifier.joblib')
    
    PATH_STATIC = os.path.join(PATH_BASE, 'app', 'static')
    
    PATH_ATTENDANCE = os.path.join(PATH_STATIC, 'attendance')
    PATH_DATASET = os.path.join(PATH_STATIC, 'dataset')
    PATH_EMBEDDING = os.path.join(PATH_STATIC, 'embedding')
    PATH_MODEL = os.path.join(PATH_STATIC, 'model')
    PATH_REGISTRATION = os.path.join(PATH_STATIC, 'registration')
    
    PATH_DATASET_TRAIN = os.path.join(PATH_DATASET, 'train')
    PATH_DATASET_VAL = os.path.join(PATH_DATASET, 'val')
    
    PATH_EMBEDDING_TRAIN = os.path.join(PATH_EMBEDDING, 'train.npz')
    PATH_EMBEDDING_VAL = os.path.join(PATH_EMBEDDING, 'val.npz')
    
    PATH_ENCODER = os.path.join(PATH_MODEL, 'encoder.joblib')
    PATH_CLASSIFIER = os.path.join(PATH_MODEL, 'classifier.joblib')
    
    os.makedirs(PATH_REGISTRATION, exist_ok=True)
    os.makedirs(PATH_MODEL, exist_ok=True)
    os.makedirs(PATH_EMBEDDING, exist_ok=True)
    os.makedirs(PATH_DATASET, exist_ok=True)
    os.makedirs(PATH_DATASET_TRAIN, exist_ok=True)
    os.makedirs(PATH_DATASET_VAL, exist_ok=True)
    os.makedirs(PATH_ATTENDANCE, exist_ok=True)
    os.makedirs(PATH_BASE, exist_ok=True)
    