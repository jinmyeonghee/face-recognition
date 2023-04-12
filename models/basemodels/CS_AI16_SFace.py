import os, gdown
import tensorflow as tf
from .SFace import loadModel as load_model

script_dir = os.path.dirname(os.path.abspath(__file__))

def loadModel():
    model = load_model()



import tensorflow as tf

def create_siamese_network(base_model):
    # 입력 이미지의 크기 설정 (예: 160x160x3)
    input_shape = (160, 160, 3)

    # 샴 네트워크의 두 입력 이미지 정의
    input_img1 = tf.keras.layers.Input(shape=input_shape)
    input_img2 = tf.keras.layers.Input(shape=input_shape)

    # 특징 추출기로 사용할 사전 학습된 모델 로드
    feature_extractor = loadModel()

    # 두 이미지의 특징 벡터 계산
    features_img1 = feature_extractor(input_img1)
    features_img2 = feature_extractor(input_img2)

    # 사용자 정의 유클리디안 거리 계산 레이어
    class EuclideanDistanceLayer(tf.keras.layers.Layer):
        def call(self, feature_vectors):
            feat_vector1, feat_vector2 = feature_vectors
            sum_squared = tf.math.reduce_sum(tf.math.square(feat_vector1 - feat_vector2), axis=1, keepdims=True)
            return tf.math.sqrt(tf.math.maximum(sum_squared, tf.keras.backend.epsilon()))

    # 두 특징 벡터 사이의 유클리디안 거리 계산
    euclidean_distance = EuclideanDistanceLayer()([features_img1, features_img2])

    # 유사성 점수 출력을 위한 시그모이드 활성화 함수를 사용하는 완전 연결(Dense) 레이어
    similarity_output = tf.keras.layers.Dense(1, activation='sigmoid')(euclidean_distance)

    # 최종 샴 네트워크 모델 정의
    siamese_network = tf.keras.Model(inputs=[input_img1, input_img2], outputs=similarity_output)

    return siamese_network

# 샴 네트워크 생성
siamese_network = create_siamese_network(base_model)
