import os
import joblib
from detectors import YOLOv8nFace, MiniFASNet, Facenet512, YOLOv8mMaskOpenCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def get_facenet_512() -> Facenet512:
    global model_facenet_512

    if 'model_facenet_512' in globals():
        return model_facenet_512

    model_facenet_512 = Facenet512(
        'assets/model/facenet512_weights.onnx',
        verbose=False,
    )
    return model_facenet_512


def get_minifasnet() -> MiniFASNet:
    global model_minifasnet

    if 'model_minifasnet' in globals():
        return model_minifasnet

    model_minifasnet = MiniFASNet(
        'assets/model/2.7_80x80_MiniFASNetV2.onnx',
        verbose=False,
    )
    return model_minifasnet


def get_yolov8n_face() -> YOLOv8nFace:
    global model_yolov8n_face

    if 'model_yolov8n_face' in globals():
        return model_yolov8n_face

    model_yolov8n_face = YOLOv8nFace(
        'assets/model/yolov8n-face-hpc203.onnx',
        verbose=False,
    )
    return model_yolov8n_face


def get_yolov8m_mask() -> YOLOv8mMaskOpenCV:
    global model_yolov8m_mask

    if 'model_yolov8m_mask' in globals():
        return model_yolov8m_mask

    model_yolov8m_mask = YOLOv8mMaskOpenCV(
        'assets/model/yolov8m-mask.onnx',
        verbose=False,
    )
    return model_yolov8m_mask


def get_encoder(model_updated='static/model/encoder.joblib') -> LabelEncoder:
    if os.path.exists(model_updated):
        return joblib.load(model_updated)
    return joblib.load('assets/model/encoder.joblib')


def get_classifier(model_updated='static/model/classifier.joblib') -> LabelEncoder:
    if os.path.exists(model_updated):
        return joblib.load(model_updated)
    return joblib.load('assets/model/classifier.joblib')
