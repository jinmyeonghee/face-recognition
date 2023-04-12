import glob
import cv2
import os
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

from .function.get_embedding import *
from .function.get_similarity import *


def pipeline(input_base_path):
    """
    함수목적 : db에서 로드한 사진에서 얼굴 이미지 처리 과정을 거쳐 나온 사진의 임베딩값을 추출하고,
            유사도를 구하는 파이프라인
    인풋 : input_base_path(리사이징까지 마친 이미지 폴더) 
    아웃풋 정보 : 유사도
    예외처리 : 얼굴 못찾으면 return None 
    """
    # 모델 크기 딕셔너리
    TARGET_SIZES = {
    "VGGFace": (224, 224),
    "Facenet": (160, 160),
    "Facenet512": (160, 160),
    "OpenFace": (96, 96),
    "FbDeepFace": (152, 152),
    "DeepID": (55, 47),
    "Dlib": (150, 150),
    "ArcFace": (112, 112),
    "SFace": (112, 112),
    }


    # 모델 로드 # 사용할 모델이름과 metric 작성 ######
    model_name = "VGGFace"         # 위의 타겟사이즈에 있는 모델 중 하나
    distance_metric = "euclidean"     # "cosine", "euclidean", "euclidean_l2"

    target_size = TARGET_SIZES[model_name]

    image_paths = glob.glob(os.path.join(input_base_path, '*resized.png'))

    embedding_dict = get_face_embedding_dict(image_paths, model_name, target_size)

    name1 = '/Users/jinmh/Desktop/face2/b1_000_04resized'
    name2 = '/Users/jinmh/Desktop/face2/z2_000_04resized'

    get_distance(name1, name2, model_name, distance_metric, embedding_dict)


# input_base_path(리사이징까지 마친 이미지 폴더) 
input_base_path = '/Users/jinmh/Desktop/face2'
pipeline(input_base_path)