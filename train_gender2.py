# pip install -q h5py pyyaml    # 체크포인트 사용 위해 설치 필요

import tensorflow as tf

from deepface.commons import functions

tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import Convolution2D, Flatten, Activation
elif tf_version == 2:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation

####################################################################################
# GPU
# NVIDIA 그래픽 카드가 있는 경우
import cv2  # 이 부분 아니어도 밑에서 필요
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    cv2.cuda.setDevice(0)

# time
from tqdm import tqdm
import time

## 데이터셋 불러오기
import pandas as pd
import os
import openpyxl

import numpy as np
from utils.face_detector import FacePreparer
from utils.function.generals import load_image

from sklearn.model_selection import train_test_split



n = 1               # 수정 -> (n = 파일명 번호 - 1)
init = 100 * n
start = init + 1    
end = 100 * (n+1)  # 뒤의 숫자 분할 학습 숫자 맞춰서 변경 

strat_time = time.time()
# dataset 이미지 경로 파일
excel_path = "C:/Users/userpc/handa/cp2/AI_Hub_gender_with_path_sheet.xlsx"
img_base_path = 'C:/Users/userpc/Downloads/High_Resolution/'

# 데이터 엑셀 시트 데이터프레임으로 합치는 코드
def get_label_data(excel_path, img_path):
    """ 
    id(400) - gender(400) / 이미지 경로(4302) 컬럼으로 시트가 나눠진 엑셀 파일에서 데이터프레임을 가져온다
    Args:
        input: 엑셀 경로, 이미지 파일 경로
        output: ID - Gender - realpath 컬럼을 가지는 데이터프레임
    """
    excel = pd.read_excel(excel_path, sheet_name=None, engine='openpyxl')
    df_path = excel['path']
    df_id = pd.DataFrame()
    df_id0_label = excel['Sheet'].iloc[:, :2].astype('int')
    df_id_label = excel['Sheet'].iloc[:, :2].astype('int')
    df_id0_label['realpath'] = df_id0_label['ID'].apply(lambda x: os.path.join(img_path, str(x), df_path['path'][init]))
    for i in range(start, end):    # 400*100 = 40,000개 path 생성
        df_id_label['realpath'] = df_id0_label['ID'].apply(lambda x: os.path.join(img_path, str(x), df_path['path'][i]))
        if i == start:
            df_id = pd.concat([df_id0_label, df_id_label], ignore_index=True)
        else:
            df_id = pd.concat([df_id, df_id_label], ignore_index=True)
    for i in range(len(df_id)):    # 경로 기호 수정
        df_id.at[i,'realpath'] = df_id.at[i, 'realpath'].replace('\\','/')
    
    # change gender labeling - (0: 남성, 1: 여성) -> (0: 여성, 1: 남성)
    label_map = {0:1, 1:0}
    df_id['Gender'] = df_id['Gender'].apply(lambda x: label_map[x])

    return df_id

## dataset
df = get_label_data(excel_path, img_base_path)

print("load excel")


## datset - df[realpath(img_array), gender]
model_name = 'vggface'
preparer = FacePreparer()
current_path = os.getcwd() # 주피터 노트북 위치

def nparray_set_maker(x):
    img = load_image(x, current_path)
    img_array = preparer.detect_faces(img, model_name) # [w, h, 3] 배열이나, 빈 list []를 리턴함
    if img_array is None or len(img_array) == 0:
        img_array = np.nan
    else:
        img_array = img_array[0]
    return img_array


### Generator를 통해 float으로 변경
## Generator 사용을 위한 함수 생성
# Dataframe 각 열을 리스트로 반환
realpath_list = df['realpath'].tolist()
gender_list = df['Gender'].tolist()

# Dataframe으로 부터 TF Dataset 생성하는 함수 - 추출되지 않은 값은 해당 데이터를 건너뛰고 다음 데이터 생성(기존 copy(),dropna()과정)
def gen():
    for i in range(len(df)):
        realpath = nparray_set_maker(realpath_list[i])
        if np.isnan(realpath).any():
            continue  # np.nan 값이 있는 경우, 해당 데이터를 건너뛰고 다음 데이터 생성

        yield(
            realpath / 255.0,   # 정규화
            gender_list[i]      # label(output)
        )

df_train_raw = tf.data.Dataset.from_generator(gen, output_types=(tf.float64, tf.int16))

print("load image from path & resize images")

# 데이터 크기가 너무 배치 단위로 나눠줌
df_train_batch = df_train_raw.batch(1000)

for batch, (x,y) in enumerate(df_train_batch):
    pass

## Split Train/Val/Test set
# data = np.stack(df_train_dropped['realpath'])   # data (144000-drop, 224, 224, 3)
data = np.array(x)        # (1000, 224, 224, 3)     # batch_size: 1000
target = np.array(y)      # label (1000,)           # batch_size: 1000

print('data.shape :', data.shape, ' target.shape :', target.shape)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=True, stratify=target, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print('Train :', X_train.shape, y_train.shape, ' Valid :', X_val.shape, y_val.shape, ' Test :', X_test.shape, y_test.shape)

end_time = time.time()
print("prepare time(sec): ", round(end_time - strat_time,1))

##############################################################################################################

# Labels for the genders that can be detected by the model.
# labels = ["Woman", "Man"]

## Model
from utils.gender_distinguisher import GenderDistinguisher
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

model = GenderDistinguisher().model

print("load model")


## 모델 학습 - transfer learning fine tuning을 통한 AI_Hub 데이터 학습
# 컴파일 후, 학습 진행
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0001),
                  loss = 'sparse_categorical_crossentropy',        # activation: softmax
                  metrics=['accuracy'])
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)    
strat_time = time.time()
hist = model.fit(X_train, y_train, epochs = 100, validation_data=(X_val, y_val), batch_size= 32, callbacks=[early])
end_time = time.time()
print("n epochs model fit time(sec): ", round(end_time - strat_time,1))
print("Complete train model")

# 학습 과정 시각화
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

file_name = "gender_train_weights" + str(n+1)
plt.savefig(f'./models/weights/batch40000/{file_name}.png')  

## Accuracy 측정 - 최종 test set을 이용한 evaluate은 학습이 모두 끝난 후 한 번만 진행하는 것이 맞으나, 현재 데이터셋이 다 분할되어 있으므로 진행
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_acc)

# 가중치 저장
model.save_weights(f"./models/weights/batch40000/{file_name}.h5")  # 불러다 evaluate(테스트셋 성능 평가)은 가능하나, 이 아이로 이어서 추가학습을 시킬 수 있는 건 아닌 것 같음
