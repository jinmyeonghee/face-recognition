from pathlib import Path
import argparse
import sys
import os, math
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
from utils.math import get_contrastive_loss
from utils.plot import plot_history, pair_plot, cm_plot
from utils.trainData import *
from utils.face_verifier2 import Verifier2


def build_model(model_name, distance_metric):
    """학습 모델 구성, x: 이미지 두 장, y: id, gender
    input: 
        기본 모델 이름
    output: 
        동일인 여부 예측 모델
    """
    verifier = Verifier2(model_name, distance_metric)
    return verifier.model
# -----------------------------------

# def create_df_image_paths(row):
#     paths = df_paths['path'].apply(lambda x: os.path.join(row, x))
#     return pd.DataFrame(paths)


def train(df_path, img_path, model_name="vggface", distance_metric='cosine', batch_size=32, epochs=3, optimizer='adam', lr=0.001):
    """모델 학습 함수
    input: 
        df_path : 학습시킬 이미지와 라벨 정보가 들어있는 엑셀파일 경로
        img_path : 이미지파일들이 저장되어 있는 상위 폴더 경로
        model : 가져올 모델 이름
        batch_size : 작게 설정할수록 학습에 시간이 더 오래 걸리지만, 메모리는 더 적게 쓸 수 있음
    output: 
        학습결과
    """
    set_seeds() # 시드 고정
    dir_path = "./weights"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # Dataset 준비 -----------------
    n_test = 10000 # 테스트용 ######################

    start = time.time()
    # 학습셋 정보 (이미지경로 및 라벨(image_path-id-gender) 엑셀) 읽어오기
    df = get_label_data(df_path)  
    df = df.sample(n=n_test, random_state=42) # 테스트용으로 일부 data 추출
    print("-> get_label_data time: ", time.time()-start)

    # train, test 데이터셋 split  -> (1376640,) (344160,)
    train, test = train_test_split( 
        df, test_size=0.2, stratify=df['Gender'], random_state=42)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    print("train, test shape: ", train.shape, test.shape)
    # print("X_train, X_test, y_train, y_test shape: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    train_generator = generator(train, img_path, model_name, batch_size)
    val_generator = generator(test, img_path, model_name, batch_size)

    # # 스텝 설정
    # steps_per_epoch = math.ceil(len(train) * NUM_PER_ID * 2 / batch_size)
    # validation_steps = math.ceil(len(test) * NUM_PER_ID * 2 / batch_size)

    run_options = tf.compat.v1.RunOptions()
    run_options.report_tensor_allocations_upon_oom = True



    # 모델 준비 -----------------

    # 학습할 모델 불러오기
    model = build_model(model_name, distance_metric) 
    # model = load_model('VGG-Face') # 테스트

    # 손실 함수, 평가 지표 정의
    loss = ["binary_crossentropy"] 
    metrics = ["accuracy"]
    
    # optimizer 정의
    if optimizer.upper() == 'ADAGRAD':
        opt = tf.keras.optimizers.Adagrad(learning_rate=lr)
    elif optimizer.upper() == 'ADADELTA':
        opt = tf.keras.optimizers.Adadelta(learning_rate=lr, rho=0.9, epsilon=1e-6)
    elif optimizer.upper() == 'ADAM':
        opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    elif optimizer.upper() == 'RMSPROP':
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=0.9, momentum=0.9, epsilon=1.0)
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
    ## LambdaCallback 설정
    keep_latest_weights_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: keep_latest_n_weights(dir_path, n=10)
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='weights/' + model_name + '_weights_epoch_{epoch:02d}-{accuracy:.3f}.h5',
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=False,
        save_freq='epoch', # 1에폭마다 저장함
        verbose=1,
    )

    print(f'---------- fit model ----------')
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping_callback, keep_latest_weights_callback],
        batch_size = batch_size,
        verbose=1,
        steps_per_epoch=len(train) // batch_size,
        validation_steps= len(test) // batch_size
    )

    # 모델 평가 (나중에 main 함수 짜고 정리필요)
    plot_history(history)
    print('---------- evaluate model ----------')
    results = model.evaluate(val_generator)
    # 예측
    # print('---------- predict test set ----------')
    # predictions = model.predict([pairImgTest[:, 0], pairImgTest[:, 1]])
    # # 테스트 쌍 예측 결과 시각화
    # pair_plot(pairImgTest, pairIdTest, pairSexTest, to_show=21, predictions=predictions, test=True)
    # cm_plot(pairIdTest, predictions[0], 'face')

    
    # # best weight 불러오기
    # model.load_weights(save_path)
    
    return history, results


def keep_latest_n_weights(dir_path, n=10):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        return
    
    weights_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.h5')], 
                           key=lambda x: os.path.getmtime(os.path.join(dir_path, x)), reverse=True)
    for f in weights_files[n:]:
        os.remove(os.path.join(dir_path, f))




data_path = '../make_traindata/id-gender-img_path.xlsx'
img_path = '../DATA_AIHub/dataset/'

# train_dataset, val_dataset = train(data_path, img_path)
# print("train_dataset[0](Img), train_dataset[1](Label) shape:",train_dataset[0].shape, train_dataset[1].shape)
# print("val_dataset[0](Img), val_dataset[1](Label) shape:",val_dataset[0].shape, val_dataset[1].shape)

history, val_result = train(data_path, img_path, batch_size=8)
print(history)
print()
print(val_result)















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