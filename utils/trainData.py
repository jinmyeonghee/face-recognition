import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.generals import *



def get_label_data(df_path):
    """ 이미지 경로 - id - gender 컬럼을 가지는 데이터프레임을 가져온다
    Args:
        input: 엑셀 경로
        output: File_Path - ID - Gender 컬럼을 가지는 데이터프레임
    """
    xlsx = pd.read_excel(df_path, sheet_name=None, engine="openpyxl")
    df = pd.concat(xlsx.values()) # 모든 시트 합침
    df.reset_index(drop=True, inplace=True) # 인덱스 재설정

    return df
    
# -----------------------------------


def create_pairs(df):
    """ 동일인여부 예측을 위해 이미지 쌍을 만들어주는 함수

    
    """


# -----------------------------------


def create_datasets(df_path, img_path, target_size=(224, 224), batch_size=32):
    """ 데이터를 불러오고 모델에 맞는 형태로 변환해주는 함수
    input: 
        df_path : File_Path, ID, Gender 정보를 가지는 엑셀파일 경로
        img_path : 이미지파일들이 저장되어 있는 root 폴더 경로
    output: 
        File_Path - ID - Gender 컬럼을 가지는 데이터프레임
    """
    img_base_path = img_path

    # 라벨 및 이미지경로 엑셀(image_path-id-gender) 읽어오기
    df = get_label_data(df_path) 

    # # 라벨을 숫자로 매핑 (기업데이터 해당)
    # label_map = {'0': 0, '2': 1, '남성': 0, '여성': 1} # 2:인증(동일인)
    # df['labeled_result_value'] = df['labeled_result_value'].apply(lambda x: label_map[x])
    # df['gender'] = df['gender'].apply(lambda x: label_map[x])

    # 데이터프레임 데이터를 변수로 저장
    img_sets = []
    for im in df['File_Path']:  
        img_sets.append(load_image(str(Path(os.path.join(img_base_path,im)))))
    img_sets = np.array(img_sets) # list -> np.ndarray (데이터수, 열, 행, rgb)
    id_sets = df['ID'].values  # pd.Series -> np.ndarray
    gender_sets = df['Gender'].values  # pd.Series -> np.ndarray

    # train, test 데이터셋 split
    X_train, X_test, y_train, y_test, y_g_train, y_g_test = train_test_split(
        img_sets, id_sets, gender_sets, test_size=0.2, random_state=42)




    
    # 긍정, 부정 쌍으로 이루어진 데이터셋으로 만들기
    
    

    
    # 이미지 경로 -> np.array = load_image 
    # -> train/test split
    # -> 이미지쌍 - id(->동일인여부) - gender (arr)



    
    
    return train_dataset, validation_dataset
# -----------------------------------


data_path = '../DATA_AIHub/dataset/'
data_path = '../../make_traindata/id-gender-img_path.xlsx'

train_data, val_data = create_datasets(data_path)
print(type(train_data))
print(train_data)