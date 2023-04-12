'''
similarity(유사도) : 두 벡터를 비교해 검증.
벡터 비교 방법은 코사인거리, 유클리디안 거리, 유클리디안 L2거리
가장 쉬운 벡터 비교 방법은 유클리디안 거리를 찾는 것 
'''
import numpy as np
import math
import tensorflow as tf

# 코사인유사도
# def cosine_similarity(source, test):
#     source = l2_normalize(source)
#     test = l2_normalize(test)
#     cosine_similarity = np.dot(source, test)
#     return cosine_similarity


def cosine_distance(source, test):
    a = np.matmul(np.transpose(source), test)
    b = np.sum(np.multiply(source, source))
    c = np.sum(np.multiply(test, test))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def euclidean_distance(source, test):
    # 각 차원의 차이의 제곱의 합을 루트한 값
    euclidean_distance = source - test
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance
    

def euclidean_l2_distance(source, test):
    # 벡터를 정규화한 후 차원 별로 제곱한 값들의 합을 구한 뒤 루트한 값
    # L2 normalization
    source = source / np.linalg.norm(source)
    test = test / np.linalg.norm(test)
    
    # Compute L2 distance
    euclidean_l2_distance = source - test
    euclidean_l2_distance = np.sqrt(np.sum(np.square(euclidean_l2_distance)))

    return euclidean_l2_distance



def findThreshold(model_name, distance_metric):
    
    base_threshold = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}

    thresholds = {
        "VGGFace".capitalize(): {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86},
        "VGG-Face".capitalize(): {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86},
        "Facenet".capitalize(): {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
        "Facenet512".capitalize(): {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
        "ArcFace".capitalize(): {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
        "Dlib".capitalize(): {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
        "SFace".capitalize(): {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
        "OpenFace".capitalize(): {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
        "FbDeepFace".capitalize(): {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
        "DeepID".capitalize(): {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17},
    }

    if model_name.capitalize() in thresholds:
        return thresholds[model_name.capitalize()].get(distance_metric, 0.4)
    else:
        return 0.4





# 두 얼굴의 거리를 측정하여 유사도를 계산하고, 유사한 이미지인지 판단
def get_distance(origin_embedding, target_embedding, model_name, distance_metric = 'cosine'):
    if len(origin_embedding) > 0 and len(target_embedding) > 0:
        if distance_metric == "cosine":
            distance = cosine_distance(origin_embedding, target_embedding)
        elif distance_metric == "euclidean":
            distance = euclidean_distance(origin_embedding, target_embedding)
        elif distance_metric == "euclidean_l2":
            distance = euclidean_l2_distance(origin_embedding, target_embedding)
        else:
            raise ValueError("Invalid distance metric")
        
        threshold = findThreshold(model_name, distance_metric)
        # print(threshold)
        if threshold > distance:
            return { "임계값": threshold, "두 얼굴의 유사도": round(distance, 2), "일치 여부" : True }
            #return True
        else:
            return { "임계값": threshold, "두 얼굴의 유사도": round(distance, 2), "일치 여부" : False }
            #return False

    else:
       return ('두 이미지에서 얼굴을 찾을 수 없습니다.', False)

