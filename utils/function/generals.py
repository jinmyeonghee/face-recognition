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

    target_size = target_sizes.get(model_name)

    if target_size == None:
        raise ValueError(f"unimplemented model name - {model_name}")

    return target_size