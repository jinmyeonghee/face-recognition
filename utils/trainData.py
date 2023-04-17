import os
import pandas as pd
import numpy as np
import cv2
import random
from tqdm import tqdm
from pathlib import Path
import gc
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


def img_transform(img_path, img_base_path, model_name):
    """ batch_size 단위로 이미지 경로를 array로 읽어오고 
        얼굴 부분만 크롭, 패딩, 리사이즈의 이미지 전처리를 수행
    input: 
        img_path : 이미지 경로 str 
        img_base_path : 이미지파일들이 저장되어 있는 상위 폴더 경로
        model : 사용하는 모델 이름
        batch_size : 한 번에 처리할 데이터 수
    output:
        이미지를 (데이터 수, target_size[0], target_size[1], 채널수) 형태의 np.ndarray로 반환
    """
    preparer = FacePreparer()
    img_sets = []
    
    img = cv2.imread(str(Path(os.path.join(img_base_path, img_path))))
    faces = preparer.detect_faces(img, model_name, align=False)  # 얼굴 탐지, 크롭, 패딩, 리사이즈
    if len(faces) > 0: 
        img = faces[0]/255.0 # 이미지 리스트에서 감지된 얼굴 하나 선택 & 정규화
    else:
        img = None
        # print("얼굴이 검출되지 않았습니다 -> ", path)

    return img  # array([[[ ]]]) / None
# -----------------------------------


def generator(df, base_path, model_name, generator_batch= 32): #클래스에 대해 반복
    """ 동일인여부 예측을 위해 긍정/부정 이미지 쌍을 만들어주는 함수
    input: 
        이미지 list, id array를 입력
    output: 
        ((이미지, 이미지), 동일인여부 라벨)
    """
    X = df["File_Path"]
    y = df["ID"]

    # print("Creating image pairs ...")
    pair_source_list = [] 
    pair_target_list = []
    pair_label_list = [] # 긍정(두 사진이 동일인):1, 부정(두 사진이 비동일인):0 레이블
    unique_labels = np.unique(y) # id 클래스
    random_unique_labels = np.random.choice(unique_labels, size=50, replace=False)


    while True:
        # 모든 클래스에 대해 반복
        for label in random_unique_labels:
            if int(np.where(random_unique_labels==label)[0]+1) > len(random_unique_labels)-1:  # 마지막 label 실행X
                return
            print(f"\tlabel:{label} ({int(np.where(random_unique_labels==label)[0]+1)}/{len(random_unique_labels)})")
            gc.collect()
            pos_indices = np.where(y == label)[0]  # id가 같은 데이터 인덱스들
            neg_indices = np.where(y != label)[0]
            n_samples = len(pos_indices) if len(pos_indices)<len(neg_indices) else len(neg_indices)
            iter_n = min(n_samples//2, 500)
            
            for i in range(iter_n):
                img_idx_1 = pos_indices[i*2]  # 긍정 짝수번째의 인덱스
                img_idx_2 = pos_indices[i*2+1]  # 긍정 홀수번째의 인덱스
                img_idx_3 = neg_indices[i*2] # 부정 인덱스
                # 이미지 배열 읽어오기
                source_img = img_transform(X[img_idx_1], base_path, model_name)
                target_img = img_transform(X[img_idx_2], base_path, model_name)
                neg_target_img = img_transform(X[img_idx_3], base_path, model_name)
                if (source_img is None) or (target_img is None) or (neg_target_img is None): 
                    # print("img pair skip - ", y[img_idx_1], source_img is None,", ", y[img_idx_2], target_img is None,", ", y[img_idx_3], neg_target_img is None)
                    continue # 얼굴이 검출되지 않은 경우 다음으로 (=사진 제외)
                # 동일 id들의 인덱스들에서 긍정쌍 생성
                pair_source_list.append(source_img)
                pair_target_list.append(target_img)
                pair_label_list.append(1)
                # 다른 id들의 인덱스와 부정쌍 생성
                pair_source_list.append(source_img)
                pair_target_list.append(neg_target_img)
                pair_label_list.append(0)
                if len(pair_label_list) >= generator_batch: # generator_batch 이상이 되면 
                    pair_source_array = np.array(pair_source_list, dtype=np.float32)
                    pair_source_list.clear()
                    pair_target_array = np.array(pair_target_list, dtype=np.float32)
                    pair_target_list.clear()
                    pair_label_array = np.array(pair_label_list, dtype=np.uint8)
                    pair_label_list.clear()
                    # print("pair_source_array, pair_target_array, pair_label_array shape: ", pair_source_array.shape, pair_target_array.shape, pair_label_array.shape)
                    yield [pair_source_array, pair_target_array], pair_label_array
        


                    