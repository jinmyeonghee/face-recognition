import os
import pandas as pd
import numpy as np
import cv2
import random
from tqdm import tqdm
from pathlib import Path
from utils.function.generals import *
from utils.face_detector import FacePreparer
import tkinter as tk
from tkinter import filedialog

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
                img = faces[0] # 이미지 리스트에서 감지된 얼굴 하나 선택
            else:
                img = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)  # 검출되지 않은 경우, 빈 이미지 생성
                # print("얼굴이 검출되지 않았습니다 -> ", img_path)
            batch_imgs.append(img)
        img_sets.append(np.array(batch_imgs))

    return np.vstack(img_sets)
# -----------------------------------


def create_pairs(X, y, batch_size=32, shuffle=True): #클래스에 대해 반복
    """ 동일인여부 예측을 위해 긍정/부정 이미지 쌍을 만들어주는 함수
    input: 
        이미지 array, id array를 입력
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
            # 동일 id들의 인덱스들에서 반복 (긍정쌍 생성)
            img_idx_1 = pos_indices[i*2]  # 짝수번째의 인덱스
            img_idx_2 = pos_indices[i*2+1]  # 홀수번째의 인덱스
            pairImages.append([X[img_idx_1], X[img_idx_2]])
            pairLabels.append(1)
            # 다른 id들의 인덱스와 반복 (부정쌍 생성)
            img_idx_2 = neg_indices[i*2]
            pairImages.append([X[img_idx_1], X[img_idx_2]])
            pairLabels.append(0)
    
    pairImages = np.array(pairImages, dtype=np.uint8) # 여기서 메모리 문제
    pairLabels = np.array(pairLabels, dtype=np.uint8)

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



def get_df_from_excel(excel_path = None, image_data_root = None, reverse_gender_value = True):
    """
    2 데이터프레임을 엑셀로부터 가져옵니다. id/gender/id_image_data_root 컬럼의 데이터프레임, path컬럼의 데이터프레임을 반환합니다.
    excel_path는 엑셀파일의 위치를 지정합니다. 지정하지 않았을 경우 팝업창을 띄웁니다.
    image_data_root는 이미지 데이터의 루트 위치를 지정합니다. 지정하지 않았을 경우 팝업창을 띄웁니다.
    reverse_gender_value 는 기본값 True고, 역할은 gender 0을 1로, 1을 0으로 가져오는 것입니다.

    """
    if excel_path is None:
        excel_path = get_filename_by_gui()

    excel = pd.read_excel((excel_path), sheet_name=None) # None으로 하면 모든 시트를 가져옵니다.

    df_id_label = excel['Sheet'].iloc[:, :2]
    df_paths = excel['path']

    if reverse_gender_value:
        df_id_label['gender'] = df_id_label['gender'].apply(lambda x: 1 - x)
    
    if image_data_root is None:
        image_data_root = get_directory_by_gui()

    df_id_label['id_image_data_root'] = df_id_label['ID'].apply(lambda x: os.path.join(image_data_root, str(x)))

    return df_id_label, df_paths

