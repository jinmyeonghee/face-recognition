### 새로운 데이터에 대한 gender model 학습 가중치를 이용한 성능 평가
from glob import glob

import os
import openpyxl
import pandas as pd
import tensorflow as tf

import numpy as np
from utils.face_detector import FacePreparer
from utils.function.generals import load_image

# time
from tqdm import tqdm
import time

# GPU
# NVIDIA 그래픽 카드가 있는 경우
import cv2  # 이 부분 아니어도 밑에서 필요
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    cv2.cuda.setDevice(0)

strat_time = time.time()
## New Data
# dataset 이미지 경로 파일
excel_path = "C:/Users/userpc/handa/new_data_gender.xlsx"

df = pd.read_excel(excel_path, engine='openpyxl')   # 6027 rows x 2 cols

# change gender labeling - (0: 남성, 1: 여성) -> (0: 여성, 1: 남성)
label_map = {0:1, 1:0}
df['gender'] = df['gender'].apply(lambda x: label_map[x])   

print("load excel")


## img_array 생성
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
img_list = df['path'].tolist()
label_list = df['gender'].tolist()

# Dataframe으로 부터 TF Dataset 생성하는 함수 - 추출되지 않은 값은 알아서 삭제(기존 copy(),dropna()과정)
def gen():
    for i in range(len(df)):
        img = nparray_set_maker(img_list[i])
        if np.isnan(img).any():
            continue  # np.nan 값이 있는 경우, 해당 데이터를 건너뛰고 다음 데이터 생성

        yield(
            img / 255.0,
            label_list[i]      # label(output)
        )

df_raw = tf.data.Dataset.from_generator(gen, output_types=(tf.float64, tf.int16))

print("load image from path & resize images")

df_batch = df_raw.batch(6027)

for batch, (x,y) in enumerate(df_batch):
    pass

# data = np.stack(df_train_dropped['realpath'])   # data (144000-dropna, 224, 224, 3)
test_data = np.array(x)     # (5975, 224, 224, 3)
test_target = np.array(y)   # (5975,)

print('data.shape :', test_data.shape, ' target.shape :', test_target.shape)

end_time = time.time()
print("prepare time(sec): ", round(end_time - strat_time,1))

########################################################################################

strat_time = time.time()

## Model
from utils.gender_distinguisher import GenderDistinguisher
import tensorflow as tf

model = GenderDistinguisher().model

# compile
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0001),
                  loss = 'sparse_categorical_crossentropy',        # activation: softmax
                  metrics=['accuracy'])

# Loads the weights
# model.load_weights('./models/weights/gender_training/gender_train_weights18.h5')
# model.load_weights('./models/weights/merged_avg_gender_weights.h5')
# model.load_weights('./models/weights/merged_weights.h5')

# Re-evaluate the model
loss, acc = model.evaluate(test_data, test_target, verbose=2)
print("New Data Train Gender Model, accuracy: {:5.2f}%".format(100 * acc), "loss: {:.4f}".format(loss))

end_time = time.time()
print("evaluate time(sec): ", round(end_time - strat_time,1))