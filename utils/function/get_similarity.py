'''
similarity(유사도) : 두 벡터를 비교해 검증.
벡터 비교 방법은 코사인거리, 유클리디안 거리, 유클리디안 L2거리
가장 쉬운 벡터 비교 방법은 유클리디안 거리를 찾는 것 
'''
import numpy as np

# 코사인유사도
# def cosine_similarity(source, test):
#     source = l2_normalize(source)
#     test = l2_normalize(test)
#     cosine_similarity = np.dot(source, test)
#     return cosine_similarity


# cosine거리
def cosine_distance(source, test):
    a = np.matmul(np.transpose(source), test)
    b = np.sum(np.multiply(source, source))
    c = np.sum(np.multiply(test, test))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


# euclidean거리
def euclidean_distance(source, test):
    # 각 차원의 차이의 제곱의 합을 루트한 값
    euclidean_distance = source - test
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance
    
# euclidean-l2거리
def euclidean_l2_distance(source, test):
    # 벡터를 정규화한 후 차원 별로 제곱한 값들의 합을 구한 뒤 루트한 값
    # L2 normalization
    source = source / np.linalg.norm(source)
    test = test / np.linalg.norm(test)
    
    # Compute L2 distance
    euclidean_l2_distance = source - test
    euclidean_l2_distance = np.sqrt(np.sum(np.square(euclidean_l2_distance)))

    return euclidean_l2_distance

#--------

def findThreshold(model_name, distance_metric):
    
    base_threshold = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}

    thresholds = {
        "VGGFace".lower(): {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86},
        "Facenet".lower(): {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
        "Facenet512".lower(): {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
        "ArcFace".lower(): {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
        "Dlib".lower(): {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
        "SFace".lower(): {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
        "OpenFace".lower(): {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
        "FbDeepFace".lower(): {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
        "DeepID".lower(): {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17},
    }

    if model_name.lower() in thresholds:
        return thresholds[model_name.lower()].get(distance_metric, 0.4)
    else:
        return 0.4





# 두 얼굴의 거리를 측정하여 유사도를 계산하고, 유사한 이미지인지 판단(임베딩 딕트 인풋)
def get_distance(name1, name2, model_name, distance_metric, embedding_dict):
    if len(embedding_dict[name1]) > 0 and len(embedding_dict[name2]) > 0:
        if distance_metric == "cosine":
            distance = cosine_distance(embedding_dict[name1], embedding_dict[name2])
        elif distance_metric == "euclidean":
            distance = euclidean_distance(embedding_dict[name1], embedding_dict[name2])
        elif distance_metric == "euclidean_l2":
            distance = euclidean_l2_distance(embedding_dict[name1], embedding_dict[name2])
        else:
            raise ValueError("Invalid distance metric")
        
        threshold = findThreshold(model_name, distance_metric)
        # print(threshold)
        if threshold > distance:
            print( { "임계값": threshold, "두 얼굴의 유사도": round(distance, 2), "일치 여부" : True })
            #return True
        else:
            print( { "임계값": threshold, "두 얼굴의 유사도": round(distance, 2), "일치 여부" : False })
            #return False

    else:
       print('두 이미지에서 얼굴을 찾을 수 없습니다.')

# 두 얼굴의 거리를 측정하여 유사도를 계산하고, 유사한 이미지인지 판단(단일 임베딩 인풋)
def calculate_distance(origin_embedding, target_embedding, model_name, distance_metric = 'cosine'):
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
        return round(distance, 2), distance < threshold

    else:
       return ('두 이미지에서 얼굴을 찾을 수 없습니다.', False)