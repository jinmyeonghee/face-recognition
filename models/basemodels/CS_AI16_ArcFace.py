import tensorflow as tf
import os
from .ArcFace import loadModel
from utils.math import get_layer
from utils.function.generals import find_target_size
from .function.network_maker import get_transformed_base_model

script_path = os.path.abspath(__file__)

def loadSiameseModel(distance_metric='cosine'):
    # 입력 이미지의 크기 설정 (예: 224x224x3)
    input_shape = find_target_size('ArcFace') + (3,) # (w, h, 3)

    # 샴 네트워크의 두 입력 이미지 정의
    input_img1 = tf.keras.layers.Input(shape=input_shape)
    input_img2 = tf.keras.layers.Input(shape=input_shape)

    base_model = loadModel()

    feature_extractor = get_transformed_base_model(base_model)
    # 두 이미지의 특징 벡터 계산
    features_img1 = feature_extractor(input_img1)
    features_img2 = feature_extractor(input_img2)


    # 사용자 정의 유클리디안/유클리디안L2/코사인 거리 계산 레이어
    distance = get_layer(features_img1, features_img2, distance_metric)

    # 유사성 점수 출력을 위한 시그모이드 활성화 함수를 사용하는 완전 연결(Dense) 레이어
    similarity_output = tf.keras.layers.Dense(1, activation='sigmoid')(distance)

    # 최종 샴 네트워크 모델 정의
    siamese_network = tf.keras.Model(inputs=[input_img1, input_img2], outputs=similarity_output)

    loadWeight(siamese_network, distance_metric)

    return siamese_network
    
def loadWeight(siamese_network, distance_metric='cosine'):
    WEIGHTS_FILE_NAME = 'arcface_siamese.h5'
    
    file_path = os.path.join(script_path, 'weights', WEIGHTS_FILE_NAME)
    
    if os.path.exists(file_path): # 나중엔 이부분에 가중치 다운로드를 집어넣고 무조건 가중치를 로드할 듯.
        siamese_network.load_weights(file_path)