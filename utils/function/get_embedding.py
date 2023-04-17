import os
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

# # 모델 크기 딕셔너리
# TARGET_SIZES = {
#     "VGGFace": (224, 224),
#     "Facenet": (160, 160),
#     "Facenet512": (160, 160),
#     "OpenFace": (96, 96),
#     "FbDeepFace": (152, 152),
#     "DeepID": (55, 47),
#     "Dlib": (150, 150),
#     "ArcFace": (112, 112),
#     "SFace": (112, 112),
# }


### 모델 로드 # 사용할 모델이름으로 작성 ######
# model_name = "VGGFace"
# target_size = TARGET_SIZES[model_name]
# model = eval(model_name + '.loadModel()')


# 이미지 파일을 numpy 배열로 변환하는 함수
def load_image(image_path, target_size):
    img = Image.open(image_path)
    img = img.resize(target_size)  
    img = np.array(img)  
    return img

# 이미지 경로로부터 임베딩 벡터를 계산하는 함수
def get_embedding(image_path, model, target_size):
    img = load_image(image_path, target_size)
    embedding = fetch_embedding(model, img)
    # print(embedding)
    return embedding

# 이미지 np.ndarray(RGB)로부터 임베딩 벡터를 계산하는 함수
def fetch_embedding(model, face_image):
    embedding = model.predict(np.expand_dims(face_image / 255, axis=0)).flatten()
    # print(embedding)
    return embedding


# image_paths에 있는 이미지 파일 전체 한번에 임베딩추출
# def get_embeddings(image_paths, model, target_size):
#     embeddings = list(map(lambda path: model.predict(np.expand_dims(load_image(path, target_size), axis=0)).flatten(), image_paths))
#     return np.array(embeddings)

# image_parhs에 있는 이미지 파일 하나씩 임베딩찾고, 파일이름과 딕셔너리로 만들기
def get_face_embedding_dict(image_paths, model_name, target_size):
    embedding_dict = {}
    model_name = model_name
    model = eval(model_name + '.loadModel()')
    target_size = target_size

    for file in image_paths:
        embeddings = get_embedding(file, model, target_size)   # 얼굴 영역에서 얼굴 임베딩 벡터를 추출
        if len(embeddings) > 0:   # 얼굴 영역이 제대로 detect되지 않았을 경우를 대비
                    # os.path.splitext(file)[0]에는 이미지파일명에서 확장자를 제거한 이름이 담긴다. 
                embedding_dict[os.path.splitext(file)[0]] = embeddings
       
    return embedding_dict

