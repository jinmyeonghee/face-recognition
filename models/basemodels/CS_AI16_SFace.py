import tensorflow as tf
from .SFace import loadModelForSiameseNetwork
from utils.math import get_layer
from utils.function.generals import find_target_size

def loadSiameseModel(distance_metric='cosine'):
    """
    샴 네트워크 모델을 구성하려면 현재로서는 pytorch를 사용해야 합니다.
    이 부분은 구성하지 않도록 하겠습니다.
    """
    siamese_network = f"SFace {distance_metric}"

    return siamese_network

def loadModel():
    model = loadModelForSiameseNetwork() 
    return model