import tensorflow as tf
import numpy as np
import os
import random
from pathlib import Path
from PIL import Image
import cv2
import requests


def set_seeds(SEED=42):
  os.environ['PYTHONHASHSEED'] = str(SEED)
  os.environ['TF_DETERMINISTIC_OPS'] = '1'

  tf.random.set_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)
# --------------------------------------------------


def find_target_size(model_name):
    """Find the target size of the model.

    Args:
        model_name (str): the model name.

    Returns:
        tuple: the target size.
    """

    target_sizes = {
        "gender".lower(): (224, 224), # Gender 모델용
        "VGGFace".lower(): (224, 224), # VGG-Face 동일모델
        "VGG-Face".lower(): (224, 224), # VGG-Face 동일모델
        "Facenet".lower(): (160, 160),
        "Facenet512".lower(): (160, 160),
        "OpenFace".lower(): (96, 96),
        "DeepFace".lower(): (152, 152),
        "DeepID".lower(): (55, 47),
        "Dlib".lower(): (150, 150),
        "ArcFace".lower(): (112, 112),
        "SFace".lower(): (112, 112),
    }

    target_size = target_sizes.get(model_name.lower())

    if target_size == None:
        raise ValueError(f"unimplemented model name - {model_name}")

    return target_size
# --------------------------------------------------


def load_image(img, project_root):
    """Load image from path, url, numpy array.

    Args:
        img: a path, url, numpy array(RGB).

    Raises:
        ValueError: if the image path does not exist.

    Returns:
        numpy array: the loaded image.
    """
    # The image is already a numpy array
    if type(img).__module__ == np.__name__:
        return img

    # The image is a url
    if img.lower().startswith("http://") or img.lower().startswith("https://"):
        return np.array(Image.open(requests.get(img, stream=True, timeout=60).raw).convert("RGB"))[
            :, :, ::-1
        ]
    
    # The image is a path
    img = os.path.join(project_root, img)
    
    if os.path.isfile(img) is not True:
        raise ValueError(f"Confirm that {img} exists")
    
    img = cv2.imread(img)
    if img is None:
        img = imread_korean(img)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_img
# --------------------------------------------------

def imread_korean(file_path):
    img_array = np.fromfile(file_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img
