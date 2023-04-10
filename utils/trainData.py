import pandas as pd
from keras.preprocessing.image import ImageDataGenerator


def create_datasets(df_path, target_size=(224, 224), batch_size=32):
    """데이터를 불러오고 모델에 맞는 형태로 변환해주는 함수
    """
    # 엑셀 파일에서 데이터프레임을 읽어오기
    df = pd.read_excel(df_path)
    
    # 라벨을 숫자로 매핑 (기업데이터 해당)
    label_map = {'0': 0, '2': 1, '남성': 0, '여성': 1} # 2:인증(동일인)
    df['same_person'] = df['same_person'].apply(lambda x: label_map[x])
    df['gender'] = df['gender'].apply(lambda x: label_map[x])

    """
    # 긍정, 부정 쌍으로 이루어진 데이터셋으로 만들기
    (pairImgTrain, pairidTrain, pairsexTrain) = create_pairs(X_train, y_train, y_train_2, train_all_img, train_all_id)
    (pairImgTest, pairidTest, pairsexTest) = create_pairs(X_test, y_test, y_test_2, test_all_img, test_all_id)
    print('finish --> create pairs ')
    # 이미지 경로 -> np.array
    """


    
    # Image Preprocessing
    # ImageDataGenerator 객체 생성
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # train, validation dataset 생성
    train_dataset = datagen.flow_from_dataframe(
        dataframe=df,
        directory=data_path,
        x_col='Extracted_Path',  # column in 'dataframe' that contains the filenames
        y_col=['same_person', 'gender'],
        subset='training',  # ImageDataGenerator에서 validation_split을 설정하여 데이터 부분집합을 정의해준 것
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True,
        class_mode='binary',  # 동일인 여부를 0과 1로 분류하므로 binary class mode
        color_mode='rgb',  # rgb / grayscale
        save_to_dir=None, # 증강된 이미지를 저장할 디렉토리
        save_prefix='Aug_' # 저장된 이미지의 파일이름에 사용할 접두부호 (save_to_dir이 활성화 됐을 경우)
    )

    validation_dataset = datagen.flow_from_dataframe(
        dataframe=df,
        directory=data_path,
        x_col='Extracted_Path',
        y_col=['same_person', 'gender'],
        subset='validation',
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode='binary',  # 성별을 0과 1로 분류하므로 binary class mode
        color_mode='rgb',
        save_to_dir=None, 
        save_prefix='Aug_'
    )
    
    return train_dataset, validation_dataset
# -----------------------------------


data_path = '../DATA_AIHub/dataset/'
data_path = '../../DATA_AIHub/dataset/'

train_data, val_data = create_datasets(data_path)
print(type(train_data))
print(train_data)