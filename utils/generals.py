import tensorflow as tf
import numpy as np
import os
import random


def set_seeds(SEED=42):
  os.environ['PYTHONHASHSEED'] = str(SEED)
  os.environ['TF_DETERMINISTIC_OPS'] = '1'

  tf.random.set_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)


def find_target_size(model_name):
    """Find the target size of the model.

    Args:
        model_name (str): the model name.

    Returns:
        tuple: the target size.
    """

    target_sizes = {
        "VGG-Face": (224, 224),
        "Facenet": (160, 160),
        "Facenet512": (160, 160),
        "OpenFace": (96, 96),
        "DeepFace": (152, 152),
        "DeepID": (55, 47),
        "Dlib": (150, 150),
        "ArcFace": (112, 112),
        "SFace": (112, 112),
    }

    target_size = target_sizes.get(model_name)

    if target_size == None:
        raise ValueError(f"unimplemented model name - {model_name}")

    return target_size