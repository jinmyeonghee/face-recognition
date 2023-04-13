import tensorflow as tf
from models.basemodels import Gender_VGGFace
# from models.basemodels import (     # 자체 모델 불러오기 - 필요하면 사용
#     VGGFace,
#     OpenFace,
#     Facenet,
#     Facenet512,
#     FbDeepFace,
#     DeepID,
#     DlibWrapper,
#     ArcFace,
#     SFace,
# )
from deepface.commons import functions
# from keras.models import load_model
# from train import load_model      # train.py에서 load_model 불가 - 다른 모듈들도 불러오면서 에러 발생
tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import Convolution2D, Flatten, Activation
elif tf_version == 2:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation


####################################################################################
# time
from tqdm import tqdm
import time

## 데이터셋 불러오기
import pandas as pd
import openpyxl

strat_time = time.time()
# dataset 이미지 경로 파일
excel_path = "sample_data/sample_gender10.xlsx"

# 데이터 시트 데이터프레임으로 합치는 코드
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

print("load excel")

# image crop alignment
from deepface import DeepFace
import mediapipe as mp
import cv2
import numpy as np
from utils.face_detector import FacePreparer

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe'
]

## dataset
df = get_label_data(excel_path)

# df의 이미지 경로를 받아와 이미지 resize
img_path = []   # df image 절대 경로
face_objs = []  # face detection and alignment and resize 결과 저장 list
input_base_path = 'sample_data/'
batch_size = 1    # 한 번에 처리할 이미지 개수

for i in range(len(df)):
    img_path.append(input_base_path + df.at[i, 'File_Path'])  

print("load image path")

# # face detection and alignment and resize (224, 224) - 기존 extract_faces 함수 실행 불가
# for j in range(0, len(img_path), batch_size):
#     batch_img_path = img_path[j:j+batch_size]   # 200개의 이미지 경로 넣어줌
#     batch_face_objs = []
#     for img in batch_img_path:
#         face_obj = DeepFace.extract_faces(img_path = img,      # list 내부 dict 형태   # face_objs[0]['facial_area'] 접근  # face, facial_area, confidence
#                                                 target_size = (224, 224), 
#                                                 detector_backend = backends[5],  # 우선 mediapipe 사용
#                                                 enforce_detection=False)
       
#         # False로 인한 빈 값은 추가X
#         if face_obj is not False:
#             batch_face_objs.append(face_obj)
#     # 얼굴 인식된 이미지만 추가
#     face_objs += batch_face_objs


## CAPR - 자체 모듈 불러와서 사용
face_preparer = FacePreparer()

for j in range(0, len(img_path), batch_size):
    batch_img_path = img_path[j:j+batch_size]   # 200개의 이미지 경로 넣어줌
    batch_face_objs = []
    for path in batch_img_path: 
        img = cv2.imread(path)
        if img is None:         # 예외처리 - 이미지 파일이 존재하지 않는 경우 
            continue
        img_array = np.array(img)
        if img_array.size == 0: # 예외처리 - 이미지 파일이 존재하지 않는 경우
            continue
        face_objs.append(face_preparer.detect_faces(img_array))


print(face_objs[0][0][0])
print("----------------")
print(face_objs[0][0][1])
print("----------------")
print(face_objs[0][0][2])
print("----------------")
print(face_objs[0][0][9])
print("----------------")
print(len(face_objs))

print("create resize image - face_objs")

## cv2.error: 'cv::OutOfMemoryError'
# for i in range(len(df)//200):
#     for j in range(200*i,200*(i+1)):    # len(df)로 돌리면 사진 파일이 너무 많아서 DeepFace.extract_faces에서 계속 에러 발생 - 200까지 가능
#         img_path.append(input_base_path + df.at[j, 'File_Path'].replace('\\','/'))    # 'C:/Users/userpc/Downloads/High_Resolution/19062421\\S001\\L1\\E01\\C15.jpg'
#         # face detection and alignment and resize (224, 224)
#         face_obj = DeepFace.extract_faces(img_path = img_path[j],      # list 내부 dict 형태   # face_objs[0]['facial_area'] 접근  # face, facial_area, confidence
#                 target_size = (224, 224), 
#                 detector_backend = backends[5],  # 우선 mediapipe 사용
#                 enforce_detection=False 
#         )
#         if face_obj is not False:
#             face_objs.append(face_obj)


## datset - df[img_array, gender]
dataset = pd.DataFrame(columns = ['img_array', 'gender'])

# change gender labeling - (0: 남성, 1: 여성) -> (0: 여성, 1: 남성)
label_map = {0:1, 1:0}
df['Gender'] = df['Gender'].apply(lambda x: label_map[x])

# dataset value
for i in range(len(df)):
    dataset.loc[i] = [face_objs[0][0][i], df['Gender'][i]]


## Split Train/Val/Test set
from sklearn.model_selection import train_test_split

data = dataset['img_array']     # data (1720800, 224, 224, 3)
target = dataset['gender']      # label (1720800,)

# data.shape 변경 (batch_size, height, width, channel) = (1720800, 224, 224, 3)
import numpy as np

data = np.zeros((len(df),), dtype=object)
for i in range(len(df)):
    data[i] = np.zeros((224, 224, 3))   # numpy.core._exceptions.MemoryError: Unable to allocate 1.15 MiB for an array with shape (224, 224, 3) and data type float64
data = np.stack(data, axis=0)
data = data/255.0   # data 전처리 (픽셀값 소수로 조정)
data = data.astype(np.float32)
target = target.astype(np.float32)
print(data[0])
print('data.shape :', data.shape, ' target.shape :', target.shape)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=True, stratify=target, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print('Train :', X_train.shape, y_train.shape, ' Valid :', X_val.shape, y_val.shape, ' Test :', X_test.shape, y_test.shape)

end_time = time.time()
print("prepare time(sec): ", round(end_time - strat_time,1))

#############################################################################################################


# Labels for the genders that can be detected by the model.
labels = ["Woman", "Man"]

# ## 기존 Weight
# # url="https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5"
# weight = 'gender_model_weights.h5'


## Model
model = Gender_VGGFace.loadModel() # VGGFace는 loadModel()에서 가중치를 불러다가 사용하는 것이므로 gender_model_weights.h5를 불러온 모델 자체를 사용하는 것으로 함

print("load model")

## 모델 학습 - transfer learning fine tuning을 통한 AI_Hub 데이터 학습
# 컴파일 후, 학습 진행
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0001),
                  loss = 'sparse_categorical_crossentropy',        # activation: softmax
                  metrics=['accuracy'])
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)    
strat_time = time.time()
hist = model.fit(X_train, y_train, epochs = 10, validation_data=(X_val, y_val), batch_size= 256, callbacks=[early])
end_time = time.time()
print("10 epochs model fit time(sec): ", round(end_time - strat_time,1))
print("Complete train model")

# 학습 과정 시각화
import matplotlib.pyplot as plt

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

plt.show()

## Accuracy 측정
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_acc)

# 가중치 저장
model.save_weights("./gender_train_weights.h5")


# model.summary()

# # 가중치 출력
# weights = model.get_weights()
# for i, weight in enumerate(weights):
#     print(f"Weight {i + 1}: {weight.shape}")