import os
import pandas as pd
import numpy as np
import cv2
import random
from tqdm import tqdm
from pathlib import Path
from utils.function.generals import *
from utils.face_detector import FacePreparer


def get_label_data(df_path, nrows=None):
    """ 이미지 경로 - id - gender 컬럼을 가지는 데이터프레임을 가져온다
    input: 
        File_Path, ID, Gender 정보를 가지는 엑셀파일 경로
        nrows: 가지고 올 데이터 행수 (test용)
    output: 
        File_Path - ID - Gender 컬럼을 가지는 데이터프레임
    """
    print("Data file loading ...")
    xlsx = pd.read_excel(df_path, sheet_name=None, nrows=nrows, 
                         dtype={'File_Path': str, 'ID': 'int32', 'Gender': 'int8'},
                         engine="openpyxl")
    df = pd.concat(xlsx.values()) # 모든 시트 합침
    df.reset_index(drop=True, inplace=True) # 인덱스 재설정

    return df
# -----------------------------------


def img_transform(img_path_arr, img_base_path, model, batch_size=32):
    """ batch_size 단위로 이미지 경로를 array로 읽어오고 
        얼굴 부분만 크롭, 패딩, 리사이즈의 이미지 전처리를 수행
    input: 
        img_path_arr : 이미지 경로들의 array
        img_base_path : 이미지파일들이 저장되어 있는 상위 폴더 경로
        model : 사용하는 모델 이름
        batch_size : 한 번에 처리할 데이터 수
    output:
        이미지를 (데이터 수, target_size[0], target_size[1], 채널수) 형태의 np.ndarray로 반환
    """
    preparer = FacePreparer()
    target_size = find_target_size(model)
    img_sets = []
    for i in tqdm(range(0, len(img_path_arr), batch_size)):
        batch_paths = img_path_arr[i:i+batch_size]
        batch_imgs = []
        for img_path in batch_paths:
            img = cv2.imread(str(Path(os.path.join(img_base_path, img_path))))
            faces = preparer.detect_faces(img, model, align=False)  # 얼굴 탐지, 크롭, 패딩, 리사이즈
            if len(faces) > 0: 
                img = faces[0]/255.0 # 이미지 리스트에서 감지된 얼굴 하나 선택 & 정규화
            else:
                img = "None"
                # print("얼굴이 검출되지 않았습니다 -> ", img_path)
            batch_imgs.append(img)
        img_sets.extend(np.array(batch_imgs))

    return img_sets
# -----------------------------------


def create_pairs(X, y, batch_size=32, shuffle=True): #클래스에 대해 반복
    """ 동일인여부 예측을 위해 긍정/부정 이미지 쌍을 만들어주는 함수
    input: 
        이미지 list, id array를 입력
    output: 
        ((이미지, 이미지), 동일인여부 라벨)
    """
    pairImages = []  # (이미지, 이미지) 쌍
    pairLabels = [] # 긍정(두 사진이 동일인):1, 부정(두 사진이 비동일인):0 레이블
    unique_labels = np.unique(y) # id 클래스

    # 모든 클래스에 대해 반복
    for label in unique_labels:
        pos_indices = np.where(y == label)[0]  # id가 같은 데이터 인덱스들
        neg_indices = np.where(y != label)[0]
        n_samples = len(pos_indices) if len(pos_indices)<len(neg_indices) else len(neg_indices)
        for i in range(n_samples//2):
            img_idx_1 = pos_indices[i*2]  # 긍정 짝수번째의 인덱스
            img_idx_2 = pos_indices[i*2+1]  # 긍정 홀수번째의 인덱스
            img_idx_3 = neg_indices[i*2] # 부정 인덱스
            if isinstance(X[img_idx_1], str) or isinstance(X[img_idx_2], str) or isinstance(X[img_idx_3], str): 
                print("img pair skip - ", y[img_idx_1], isinstance(X[img_idx_1], str),", ", y[img_idx_2], isinstance(X[img_idx_2], str),", ", y[img_idx_3], isinstance(X[img_idx_3], str))
                continue # 얼굴이 검출되지 않은 경우 다음으로 (=사진 제외)
            # 동일 id들의 인덱스들에서 긍정쌍 생성
            pairImages.append([X[img_idx_1], X[img_idx_2]])
            pairLabels.append(1)
            # 다른 id들의 인덱스와 부정쌍 생성
            pairImages.append([X[img_idx_1], X[img_idx_2]])
            pairLabels.append(0)
    
    pairLabels = np.array(pairLabels, dtype=np.uint8)
    pairImages = np.array(pairImages, dtype=np.uint8) # 여기서 메모리 문제

    ########
    # num_pairs = len(pairImages)
    # # 리스트를 배치 단위로 분할하여 array로 변환 후 합치기
    # pairImages_list = [pairImages[i:i+batch_size] for i in range(0, num_pairs, batch_size)]
    # pairLabels_list = [pairLabels[i:i+batch_size] for i in range(0, num_pairs, batch_size)]

    # pairImages = np.concatenate([np.array(batch, dtype=np.uint8) for batch in pairImages_list])
    # pairLabels = np.concatenate([np.array(batch, dtype=np.uint8) for batch in pairLabels_list])
    ########

    if shuffle:
        indices = np.arange(len(pairLabels))
        np.random.shuffle(indices)
        pairImages = pairImages[indices]
        pairLabels = pairLabels[indices]

    # # TensorFlow Dataset 객체 생성
    # dataset = tf.data.Dataset.from_tensor_slices((pairImages, pairLabels))
    # # preprocess pairs
    # dataset = dataset.map(lambda x, y: ((img_transform(x[0]), img_transform(x[1])), y))
    # # batch and prefetch
    # dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
    # return dataset
    return pairImages, pairLabels


