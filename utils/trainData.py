import os
import pandas as pd
import numpy as np
import cv2
import random
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.function.generals import *



def get_label_data(df_path):
    """ 이미지 경로 - id - gender 컬럼을 가지는 데이터프레임을 가져온다
    input: 
        File_Path, ID, Gender 정보를 가지는 엑셀파일 경로
    output: 
        File_Path - ID - Gender 컬럼을 가지는 데이터프레임
    """
    xlsx = pd.read_excel(df_path, sheet_name=None, engine="openpyxl")
    df = pd.concat(xlsx.values()) # 모든 시트 합침
    df.reset_index(drop=True, inplace=True) # 인덱스 재설정

    return df
# -----------------------------------


def img_transform(img_path_arr, img_base_path, target_size, batch_size=32):
    """ batch_size 단위로 이미지 경로를 array로 읽어오고 target_size로 resize
    input: 
        img_path_arr : 이미지 경로들의 array
        img_base_path : 이미지파일들이 저장되어 있는 상위 폴더 경로
        target_size : 변환할 이미지 사이즈
        batch_size : 한 번에 처리할 데이터 수
    output:
        이미지를 (데이터 수, target_size[0], target_size[1], 채널수) 형태의 np.ndarray로 반환
    """
    img_sets = []
    for i in tqdm(range(0, len(img_path_arr), batch_size)):
        batch_paths = img_path_arr[i:i+batch_size]
        batch_imgs = []
        for img_path in batch_paths:
            img = cv2.imread(str(Path(os.path.join(img_base_path, img_path))))
            img = cv2.resize(img, target_size)  # 이미지 크기를 target_size로 조정
            batch_imgs.append(img)
        img_sets.append(np.array(batch_imgs))

    return np.vstack(img_sets)
# -----------------------------------


def create_pairs(img_arr, id_arr):
    """ 동일인여부 예측을 위해 긍정/부정 이미지 쌍을 만들어주는 함수
    input: 
        이미지 array, id array를 입력
    output: 
        ((이미지, 이미지), 동일인여부 라벨)
    """
    pairImages = []  # (이미지, 이미지) 쌍
    pairLabels = [] # 긍정(두 사진이 동일인):1, 부정(두 사진이 비동일인):0 레이블

    # 모든 이미지에 대해 반복
    for ix in range(len(img_arr)):
        currentImage = img_arr[ix]
        currentID = id_arr[ix]

        try: # 현재 이미지와 같은 id를 갖는 이미지를 랜덤 선택
            posIdxs = list(np.where(id_arr == currentID))[0] 
            posIdx = random.choice(posIdxs) 
            while (posIdx == ix)&(len(posIdxs)>2): # 같은 사진을 선택하면 다시 선택
                posIdx = random.choice(posIdxs)
            
            # 긍정 쌍을 만들어 이미지와 레이블을 저장
            pairImages.append([currentImage, img_arr[posIdx]])
            pairLabels.append([1])
            
            
            # 현재 이미지와 다른 id를 갖는 이미지를 랜덤 선택
            negId = list(np.where(id_arr != currentID))
            negIdx = random.choice(negId)
            
            # 부정 쌍을 만들어 이미지와 레이블을 저장
            pairImages.append([currentImage, img_arr[negIdx]])
            pairLabels.append([0])
        
        except:
            continue
    
    return (np.array(pairImages), np.array(pairLabels))
# -----------------------------------


def create_datasets(df_path, img_path, target_size=(224, 224), batch_size=32):
    """ 데이터를 불러오고 모델에 맞는 형태로 변환해주는 함수
    input: 
        df_path : File_Path, ID, Gender 정보를 가지는 엑셀파일 경로
        img_path : 이미지파일들이 저장되어 있는 상위 폴더 경로
    output: 
        File_Path - ID - Gender 컬럼을 가지는 데이터프레임
    """
    img_base_path = img_path

    # 라벨 및 이미지경로 엑셀(image_path-id-gender) 읽어오기
    df = get_label_data(df_path) 
    df = df[:1000]

    # # 라벨을 숫자로 매핑 (기업데이터 해당)
    # label_map = {'0': 0, '2': 1, '남성': 0, '여성': 1} # 2:인증(동일인)
    # df['labeled_result_value'] = df['labeled_result_value'].apply(lambda x: label_map[x])
    # df['gender'] = df['gender'].apply(lambda x: label_map[x])

    # train, test 데이터셋 split -> (1376640,) (344160,)
    X_train, X_test, y_train, y_test = train_test_split( 
        df['File_Path'], df['ID'], test_size=0.2, stratify=df['Gender'], random_state=42)
    print("X_train, X_test, y_train, y_test shape: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # pd.Series -> np.ndarray
    if isinstance(X_train, pd.Series):
        X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values

    # 이미지 경로에서 array로 읽어오기 (메모리,시간 소요 큼)
    print("Converting image path data -> image arrays ...")
    X_train = img_transform(X_train, img_base_path, target_size, batch_size)
    X_test = img_transform(X_test, img_base_path, target_size, batch_size)

    # 긍정/부정 이미지쌍 만들기
    print("Creating image pairs ...")
    (pairImgTrain, pairLabelTrain) = create_pairs(X_train, y_train)
    (pairImgTest, pairLabelTest) = create_pairs(X_test, y_test)

    print('pairImgTrain Shape :', pairImgTrain.shape)
    print('pairLabelTrain Shape :', pairLabelTrain.shape)
    print('pairImgTest Shape :', pairImgTest.shape)
    print('pairLabelTest Shape :', pairLabelTest.shape)

    return (pairImgTrain, pairLabelTrain), (pairImgTest, pairLabelTest)
# -----------------------------------


# data_path = '../DATA_AIHub/dataset/'
# data_path = '../../make_traindata/id-gender-img_path.xlsx'
# img_path = '../../DATA_AIHub/dataset/'

# train_dataset, val_dataset = create_datasets(data_path, img_path)
# print(type(train_dataset))
# print(train_dataset)