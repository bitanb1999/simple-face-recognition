import cv2 as cv
import numpy as np
from PIL import Image


def resize_image(
        img, keep_ratio=True, height=640, width=640
) -> tuple[np.ndarray, int, int, int, int]:
    top, left, new_height, new_width = 0, 0, width, height
    if keep_ratio and img.shape[0] != img.shape[1]:
        hw_scale = img.shape[0] / img.shape[1]
        if hw_scale > 1:
            new_height, new_width = height, int(width / hw_scale)
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA)
            left = int((width - new_width) * 0.5)
            img = cv.copyMakeBorder(
                img, 0, 0, left, width - new_width - left, cv.BORDER_CONSTANT, value=(0, 0, 0)
            )  # add border
        else:
            new_height, new_width = int(height * hw_scale), width
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA)
            top = int((height - new_height) * 0.5)
            img = cv.copyMakeBorder(
                img, top, height - new_height - top, 0, 0, cv.BORDER_CONSTANT, value=(0, 0, 0)
            )
    else:
        img = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)
    return img, new_height, new_width, top, left


def make_anchors(strides, feats_hw, grid_cell_offset=0.5) -> dict:
    """Generate anchors from features."""
    anchor_points = {}
    for i, stride in enumerate(strides):
        h, w = feats_hw[i]

        x = np.arange(0, w) + grid_cell_offset  # shift x
        y = np.arange(0, h) + grid_cell_offset  # shift y

        sx, sy = np.meshgrid(x, y)
        anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
    return anchor_points


def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def softmax(x, axis=1):
    x_exp = np.exp(x)
    # Jika `x` adalah vector kolom, makak `axis=0`.
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


def align_face(img: np.ndarray, left_eye: tuple, right_eye: tuple) -> np.ndarray:
    # TODO: add feature to rotate if image flipped. so, you need key points of mouth

    # if eye could not be detected for the given image, return image itself
    if left_eye is None or right_eye is None:
        return img

    # sometimes unexpectedly detected images come with nil dimensions
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img

    angle = float(np.degrees(
        np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    ))
    img = Image.fromarray(img)
    img = np.array(img.rotate(angle))
    return img


def find_cosine_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    a = np.matmul(embedding1, embedding2)
    b = np.sum(np.multiply(embedding1, embedding1))
    c = np.sum(np.multiply(embedding2, embedding2))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def verify_embeddings(embedding1: np.ndarray, embedding2: np.ndarray) -> tuple[bool, float]:
    distance = find_cosine_distance(embedding1, embedding2)

    # --- Threshold for Facenet512
    threshold = .3

    is_verified = True if distance <= threshold else False

    return is_verified, distance


def save_image_from_array(img: np.ndarray, path: str) -> str:
    cv.imwrite(path, img)
    return path


def calculate_sharpness(image, normalize=False, max_value=1000) -> float:
    """Sharpness can be assessed using image gradients or edge detection. One common method is to use the Laplacian operator.

    Args:
        image (_type_): _description_
        normalize (bool, optional): _description_. Defaults to False.
        max_value (int, optional): _description_. Defaults to 1000.

    Returns:
        float: _description_
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
    
    if normalize:
        return min(laplacian_var / max_value, 1.0)
    return laplacian_var


def calculate_brightness(image, normalize=False) -> float:
    """Brightness is the average pixel intensity in grayscale.

    Args:
        image (_type_): _description_
        normalize (bool, optional): _description_. Defaults to False.

    Returns:
        float: _description_
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    if normalize:
        return brightness / 255.0
    return brightness


def calculate_contrast(image, normalize=False, max_value=100) -> float:
    """Contrast can be assessed by the standard deviation of pixel intensities.

    Args:
        image (_type_): _description_
        normalize (bool, optional): _description_. Defaults to False.
        max_value (int, optional): _description_. Defaults to 100.

    Returns:
        float: _description_
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    contrast = gray.std()
    
    if normalize:
        return min(contrast / max_value, 1.0)
    return contrast


def calculate_unique_intensity_levels(image, normalize=False, max_levels=256):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    unique_levels = len(np.unique(gray))
    
    if normalize:
        return unique_levels / max_levels
    return unique_levels


def calculate_shadow(image) -> float:
    """Shadow detection can be complex, but a simple approach is to look for very dark regions in the image.

    Args:
        image (_type_): _description_

    Returns:
        float: _description_
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dark_area_percentage = np.sum(gray < 50) / gray.size
    
    return 1 - min(dark_area_percentage, 1.0)


def calculate_specularity(image) -> float:
    """Specularity can be estimated by detecting bright spots.

    Args:
        image (_type_): _description_

    Returns:
        float: _description_
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    bright_spots_percentage = np.sum(gray > 200) / gray.size
    
    return 1 - min(bright_spots_percentage, 1.0)


def calculate_background_uniformity(image) -> float:
    """Assess uniformity by comparing the variance of pixel intensities in the background.

    Args:
        image (_type_): _description_

    Returns:
        float: _description_
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    background_region = gray[:int(gray.shape[0] / 4), :]  # Example region
    return 1 - min(np.std(background_region) / 255.0, 1.0)

