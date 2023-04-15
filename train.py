from pathlib import Path
import argparse
import sys
import os
import pandas as pd
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


FILE = Path(__file__).resolve() # 현 file의 경로
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.function.generals import load_image, set_seeds, find_target_size
from utils.trainData import *
from face_ds_project import FaceDSProject
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


def build_model(model, distance_metric):
    """학습 모델 구성, x: 이미지 두 장, y: id, gender
    input: 
        기본 모델 이름
    output: 
        동일인 여부 예측 모델
    """
    verifier = FaceDSProject(min_detection_confidence=0.2, model_name=model, distance_metric=distance_metric)
    target_size = find_target_size(model) # 모델에 맞는 이미지 사이즈
    
    img1 = tf.keras.layers.Input(shape = target_size)
    img2 =  tf.keras.layers.Input(shape = target_size)

    distance = verifier.verify()
    ###############################  수정 필요  ###############################

    # 마지막 레이어는 유사성 점수를 출력하기 위해 시그모이드 활성화 함수를 사용하는 단일 노드가 있는 완전 연결 레이어
    verify_outputs = tf.keras.layers.Dense(1, activation = "sigmoid", name='siamese')(distance)
    model = tf.keras.Model(inputs = [img1, img2], outputs = verify_outputs)
    return model
# -----------------------------------


def train(df_path, img_path, model="vggface", distance_metric='cosine', batch_size=32, epochs=5, optimizer='adam', lr=0.001):
    """모델 학습 함수
    input: 
        df_path : 학습시킬 이미지와 라벨 정보가 들어있는 엑셀파일 경로
        img_path : 이미지파일들이 저장되어 있는 상위 폴더 경로
        model : 가져올 모델 이름
        batch_size : 작게 설정할수록 학습에 시간이 더 오래 걸리지만, 메모리는 더 적게 쓸 수 있음
    output: 
    
    """
    set_seeds() # 시드 고정
    # save_path = Path.joinpath(Path.cwd(), Path("models"))
    save_path = os.path.join(ROOT, 'models', f'{model}-custom.hdf5') # 학습 가중치를 저장할 경로


    # Dataset 준비 -----------------

    start = time.time()
    # 학습셋 정보 (이미지경로 및 라벨(image_path-id-gender) 엑셀) 읽어오기
    df = get_label_data(df_path,10000)  # 테스트용으로 일부 data 추출 (2개시트 각각 읽어오므로 2배)
    print("-> get_label_data time: ", time.time()-start)

    # train, test 데이터셋 split  -> (1376640,) (344160,)
    X_train, X_test, y_train, y_test = train_test_split( 
        df['File_Path'], df['ID'], test_size=0.2, stratify=df['Gender'], random_state=42)
    print("X_train, X_test, y_train, y_test shape: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # pd.Series -> np.ndarray
    if isinstance(X_train, pd.Series):
        X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values

    start = time.time()
    # 이미지 경로 데이터를 array로 읽어오기 & 이미지 전처리
    print("Converting image path data -> image arrays ...")
    X_train = img_transform(X_train, img_path, model, batch_size)
    X_test = img_transform(X_test, img_path, model, batch_size)
    print("-> img_transform time: ", time.time()-start)

    start = time.time()
    # 긍정/부정 이미지쌍 만들기
    print("Creating image pairs ...")
    pairImgTrain, pairLabelTrain = create_pairs(X_train, y_train, batch_size=batch_size, shuffle=True)
    pairImgTest, pairLabelTest = create_pairs(X_test, y_test, batch_size=batch_size, shuffle=False)

    print('pairImgTrain Shape :', pairImgTrain.shape)
    print('pairLabelTrain Shape :', pairLabelTrain.shape)
    print('pairImgTest Shape :', pairImgTest.shape)
    print('pairLabelTest Shape :', pairLabelTest.shape)
    print("-> create_pairs time: ", time.time()-start)


    # 모델 준비 -----------------

    # 학습할 모델 불러오기
    # model = build_model(model, distance_metric) 
    model = load_model('VGG-Face') # 테스트

    # 손실 함수, 평가 지표 정의
    loss = [get_contrastive_loss(margin=1), "binary_crossentropy"] # contrastive / 이진분류의 대표적 손실함수 
    metrics = ["accuracy"]
    
    # optimizer 정의
    if optimizer.upper() == 'ADAGRAD':
        opt = tf.keras.optimizers.Adagrad(learning_rate=lr)
    elif optimizer.upper() == 'ADADELTA':
        opt = tf.keras.optimizers.Adadelta(learning_rate=lr, rho=0.9, epsilon=1e-6)
    elif optimizer.upper() == 'ADAM':
        opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    elif optimizer.upper() == 'RMSPROP':
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=0.9, momentum=0.9, epsilon=1e-6)
    elif optimizer.upper() == 'MOM':
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
    # history = model.fit(
    #     [pairImgTrain[i][0] for i in range(len(pairImgTrain))],
    #     [pairImgTrain[i][1] for i in range(len(pairImgTrain))],
    #     pairLabelTrain,
    #     epochs=epochs,
    #     validation_data=([pairImgTest[i][0] for i in range(len(pairImgTest))],
    #                      [pairImgTest[i][1] for i in range(len(pairImgTest))],
    #                      pairLabelTest),
    #     callbacks=[early_stopping_callback, checkpoint_callback],
    # )
    # history = model.fit(
    #     [pairImgTrain[:,0], pairImgTrain[:,1]], pairLabelTrain,
    #     epochs=epochs,
    #     validation_data=([pairImgTest[:,0], pairImgTest[:,1]], pairLabelTest),
    #     callbacks=[early_stopping_callback, checkpoint_callback]
    # )

    # # 모델 평가 (나중에 main 함수 짜고 정리필요)
    # plot_history(history)
    # print('---------- evaluate model ----------')
    # results = model.evaluate([pairImgTest[:,0], pairImgTest[:,1]], pairLabelTest)
    # # # 예측
    # # print('---------- predict test set ----------')
    # # predictions = model.predict([pairImgTest[:, 0], pairImgTest[:, 1]])
    # # # 테스트 쌍 예측 결과 시각화
    # # pair_plot(pairImgTest, pairIdTest, pairSexTest, to_show=21, predictions=predictions, test=True)
    # # cm_plot(pairIdTest, predictions[0], 'face')
    # # cm_plot(pairSexTest, predictions[1], 'gender')

    
    # # best weight 불러오기
    # model.load_weights(save_path)
    
    # return history
    return (pairImgTrain, pairLabelTrain), (pairImgTest, pairLabelTest)
# -----------------------------------



data_path = '../make_traindata/id-gender-img_path.xlsx'
img_path = '../DATA_AIHub/dataset/'

train_dataset, val_dataset = train(data_path, img_path)
print("train_dataset[0](Img), train_dataset[1](Label) shape:",train_dataset[0].shape, train_dataset[1].shape)
print("val_dataset[0](Img), val_dataset[1](Label) shape:",val_dataset[0].shape, val_dataset[1].shape)

# print(train(data_path, img_path))















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