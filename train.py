from pathlib import Path
import argparse
import sys
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


FILE = Path(__file__).resolve() # 현 file의 경로
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.function.generals import set_seeds, find_target_size
from utils.trainData import create_datasets
from utils.math import distanceLayer
from utils.math import get_contrastive_loss
from utils.plot import plot_history, pair_plot, cm_plot


from models.basemodels import (
    VGGFace,
    OpenFace,
    Facenet,
    Facenet512,
    FbDeepFace,
    DeepID,
    DlibWrapper,
    ArcFace,
    SFace,
)


def load_model(model_name):
    """기본모델을 불러오는 함수

    Args:
        model_name (str)

    Returns:
        pre-trained model
    """

    # singleton design pattern
    global model_obj

    models = {
        "VGG-Face": VGGFace.loadModel,
        "OpenFace": OpenFace.loadModel,
        "Facenet": Facenet.loadModel,
        "Facenet512": Facenet512.loadModel,
        "DeepFace": FbDeepFace.loadModel,
        "DeepID": DeepID.loadModel,
        "Dlib": DlibWrapper.loadModel,
        "ArcFace": ArcFace.loadModel,
        "SFace": SFace.loadModel,
    }

    if not "model_obj" in globals():
        model_obj = {}

    if not model_name in model_obj:
        model = models.get(model_name)
        if model:
            model = model()
            model_obj[model_name] = model
        else:
            raise ValueError(f"Invalid model_name passed - {model_name}")

    return model_obj[model_name]
# -----------------------------------
# loadModel("ArcFace")


def build_model(model):
    """학습 모델 구성, x: 이미지 두 장, y: id, gender
    input: 
        기본 모델 이름
    output: 
        동일인 여부 및 성별 예측 모델
    """
    target_size = find_target_size(model) # 모델에 맞는 이미지 사이즈
    
    # 모델 불러오기
    model = load_model(model) 

    img1 = tf.keras.layers.Input(shape = target_size)
    img2 =  tf.keras.layers.Input(shape = target_size)
    featureExtractor = load_model(model)
    featsA = featureExtractor(img1)
    featsB = featureExtractor(img2)

    # 유클리디안 거리 계산 레이어
    distance = distanceLayer()([featsA,featsB])

    # 마지막 레이어는 유사성 점수를 출력하기 위해 시그모이드 활성화 함수를 사용하는 단일 노드가 있는 완전 연결 레이어
    verify_outputs = tf.keras.layers.Dense(1, activation = "sigmoid", name='siamese')(distance)
    gender_outputs = tf.keras.layers.Dense(1, activation = "sigmoid", name='gender')(featsA) # img1 기준으로 성별 예측
    model = tf.keras.Model(inputs = [img1, img2], outputs = [verify_outputs, gender_outputs])
    return model
# -----------------------------------


def train(df_path, img_path, model="vggface", batch_size=32, epochs=5, optimizer='adam', lr=0.001):
    """모델 학습 함수
    input: 
        df_path : 학습시킬 이미지와 라벨 정보가 들어있는 엑셀파일 경로
        model : 가져올 모델 이름
    output: 
    
    """
    set_seeds() # 시드 고정
    target_size = find_target_size(model) # 모델에 맞는 이미지 사이즈
    # save_path = Path.joinpath(Path.cwd(), Path("models"))
    save_path = os.path.join(ROOT, 'models', f'{model}-custom.hdf5') # 학습 가중치를 저장할 경로

    # train, validation datasets 생성
    train_dataset, val_dataset = create_datasets(df_path, img_path, target_size, batch_size)
    # # 학습할 모델 불러오기
    model = build_model(model) 

    # 손실 함수, 평가 지표 정의
    # loss = {'verify_outputs': 'binary_crossentropy', 'gender_outputs': 'binary_crossentropy'} # 이진분류의 대표적 손실함수 / 기존기수는 binary와 contrastive 비교해 contrastive 를 이용
    # metrics = {'output1': 'binary_accuracy', 'output2': 'binary_accuracy'}
    loss = [get_contrastive_loss(margin=1), "binary_crossentropy"] # contrastive / 이진분류의 대표적 손실함수 
    metrics = ["accuracy"]
    
    # optimizer 정의
    if optimizer == 'ADAGRAD':
        opt = tf.keras.optimizers.Adagrad(learning_rate=lr)
    elif optimizer == 'ADADELTA':
        opt = tf.keras.optimizers.Adadelta(learning_rate=lr, rho=0.9, epsilon=1e-6)
    elif optimizer == 'ADAM':
        opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    elif optimizer == 'RMSPROP':
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=0.9, momentum=0.9, epsilon=1.0)
    elif optimizer == 'MOM':
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError('Invalid optimization algorithm')
    
    # 모델 컴파일
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    # callback 정의 - early stopping, model checkpointing
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                               verbose=1,  # 콜백 메세지(0:출력X or 1:출력)
                                                               patience=3)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, 
                                                             verbose=1,  
                                                             save_best_only=True, 
                                                             save_weights_only=True)
    
    # 모델 학습
    print('---------- fit model ----------')
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[early_stopping_callback, checkpoint_callback]
    )

    # 모델 평가 (나중에 main 함수 짜고 정리필요)
    plot_history(history)
    print('---------- evaluate model ----------')
    results = model.evaluate(val_dataset)
    # # 예측
    # print('---------- predict test set ----------')
    # predictions = model.predict([pairImgTest[:, 0], pairImgTest[:, 1]])
    # # 테스트 쌍 예측 결과 시각화
    # pair_plot(pairImgTest, pairIdTest, pairSexTest, to_show=21, predictions=predictions, test=True)
    # cm_plot(pairIdTest, predictions[0], 'face')
    # cm_plot(pairSexTest, predictions[1], 'gender')

    
    # best weight 불러오기
    model.load_weights(save_path)
    
    return history
# -----------------------------------



# data_path = '../make_traindata/id-gender-img_path.xlsx'
# img_path = '../DATA_AIHub/dataset/'

# train_dataset, val_dataset = train(data_path, img_path)
# print(type(train_dataset))
# print(train_dataset)
















# def run(
#     model_path = ROOT / 'models/best.onnx',
#     input = ROOT / 'data',
#     output = ROOT/ 'results',
#     conf = 0.4
# ):
#     model_path = './models/best.onnx'
#     d = Detection(model_path, conf)

#     # 해당 경로에 있는 모든 파일에 대해 detect 수행
#     for file in os.listdir(input):
#         input_path = os.path.join(input, file)
#         output_path = os.path.join(output, file)
#         d.detect(input_path, output_path)


# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', type=str, default=ROOT / 'models/best.onnx')
#     parser.add_argument('--input', type=str, default= ROOT / 'data') # 작업 진행할 폴더 경로
#     parser.add_argument('--output', type=str, default= ROOT/ 'results') # 결과가 저장될 폴더 경로
#     parser.add_argument('--conf', type=float, default= 0.4)
#     args = parser.parse_args()
#     return args


# def main(args):
#     run(**vars(args))
    

# if __name__ == '__main__':
#     args = parse_opt()
#     main(args)

# python detect.py 실행할 경우 - default값으로 정해둔 data 폴더 내 모든 파일에 대해 detect 수행
# python detection.py --onnx_path [가중치 파일 경로] --source [데이터 파일 경로] --output [결과 저장파일 경로] --conf [confidence threshold]
# ex) python detect.py --input /c/Users/LG/Desktop/test11 --output /c/Users/LG/Desktop/test11