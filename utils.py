import os
import joblib
import numpy as np

from sklearn.svm import SVC
from time import perf_counter
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

from loader import get_facenet_512


def finetune(
    src_train: str | tuple[np.ndarray, np.ndarray],
    src_val: str | tuple[np.ndarray, np.ndarray],
    dst_classifier: str,
    dst_encoder: str,
    norm=False, 
    src_aug=None,
    src_test=None,
) -> None:
    """Fine-tune and save both of classifier and encoder models.

    Args:
        src_train (str | tuple[np.ndarray, np.ndarray]): _description_
        src_val (str | tuple[np.ndarray, np.ndarray]): _description_
        dst_classifier (str): _description_
        dst_encoder (str): _description_
        norm (bool, optional): _description_. Defaults to False.
        src_aug (_type_, optional): _description_. Defaults to None.
        src_test (_type_, optional): _description_. Defaults to None.
        plot (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # --- train dataset
    if isinstance(src_train, str):
        data = np.load(src_train)
        x_train, y_train = data['x_train'], data['y_train']
    
    if isinstance(src_train, tuple):
        x_train, y_train = src_train
    
    # --- augmentation dataset
    if isinstance(src_aug, str):
        data = np.load(src_aug)
        x_train = np.concatenate([x_train, data['x_train']], axis=0)
        y_train = np.concatenate([y_train, data['y_train']], axis=0)
    
    if isinstance(src_aug, tuple):
        x_train, y_train = src_aug
    
    # --- val dataset
    if isinstance(src_val, str):
        data = np.load(src_val)
        x_val, y_val = data['x_val'], data['y_val']
    
    if isinstance(src_val, tuple):
        x_val, y_val = src_val
    
    # --- preprocessing
    if norm:
        encoder_embedding = Normalizer(norm='l2')
        x_train_scaled = encoder_embedding.transform(x_train)
        x_val_scaled = encoder_embedding.transform(x_val)
    else:
        x_train_scaled = x_train.copy()
        x_val_scaled = x_val.copy()

    encoder_y = LabelEncoder()
    y_train_scaled = encoder_y.fit_transform(y_train)
    y_val_scaled = encoder_y.transform(y_val)
    
    # --- training
    tic = perf_counter()

    param_grid = {
        'C': uniform(loc=0.1, scale=10),  # Regularization parameter
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
        'gamma': ['scale', 'auto']  # Kernel coefficient
    }

    clf = RandomizedSearchCV(
        SVC(class_weight='balanced', probability=True, random_state=42), param_grid, n_iter=50, cv=5, random_state=42
    )
    clf = clf.fit(x_train_scaled, y_train_scaled)
    
    print("done in %0.3fs" % (perf_counter() - tic))
    
    print("Best estimator found by:")
    print(clf.best_estimator_)
    
    # --- prediction
    y_pred_train = clf.predict(x_train_scaled)
    y_pred_val = clf.predict(x_val_scaled)
    
    # --- evaluation
    score_train = accuracy_score(y_train_scaled, y_pred_train)
    score_val = accuracy_score(y_val_scaled, y_pred_val)

    print('Dataset: train=%d, val=%d' % (x_train.shape[0], x_val.shape[0]))
    print('Accuracy: train=%.3f, val=%.3f' % (score_train*100, score_val*100))
    
    print('Train Classification Report')
    print(classification_report(y_train_scaled, y_pred_train, target_names=encoder_y.classes_))
    
    print('Val Classification Report')
    print(classification_report(y_val_scaled, y_pred_val, target_names=encoder_y.classes_))
    
    # --- save best models
    joblib.dump(clf.best_estimator_, dst_classifier)
    joblib.dump(encoder_y, dst_encoder)
    
    if src_test:
        if os.path.exists(src_test):
            # --- test dataset
            data = np.load(src_test)
            x_test, y_test = data['x_test'], data['y_test']
    
            # --- filter and preprocessing test dataset
            drop_ids = [i for i, _ in enumerate(y_test) if _ in set(y_test).difference(set(encoder_y.classes_))]
            x_test = [_ for i, _ in enumerate(x_test) if i not in drop_ids]
            y_test = [_ for i, _ in enumerate(y_test) if i not in drop_ids]
            
            x_test_scaled = x_test.copy()
            y_test_scaled = encoder_y.transform(y_test)
            
            y_pred_test = clf.predict(x_test_scaled)
            
            print(classification_report(y_test_scaled, y_pred_test, target_names=[_ for _ in encoder_y.classes_ if _ in y_test]))
    
    return None


def get_embeddings_dataset(data: list[np.ndarray], label: str) -> tuple[np.ndarray, np.ndarray]:
    verificator_face = get_facenet_512()
    
    x, y = [], []
    for item in data:
        embedding = verificator_face.inference(item).squeeze()
        if embedding.size == 0:
            continue
        
        x.append(embedding)
        y.append(label)
    
    return np.asarray(x), np.asarray(y)
