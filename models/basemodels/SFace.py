import os, gdown
import numpy as np
# from pathlib import Path
import cv2 as cv
# import onnx
# from onnx2torch import convert
# import tensorflow as tf
# import torch

# pylint: disable=line-too-long, too-few-public-methods
script_dir = os.path.dirname(os.path.abspath(__file__))

class _Layer:
    input_shape = (None, 112, 112, 3)
    output_shape = (None, 1, 128)


class SFaceModel:
    def __init__(self, model_path):

        self.model = cv.FaceRecognizerSF.create(
            model=model_path, config="", backend_id=0, target_id=0
        )

        self.layers = [_Layer()]

    def predict(self, image):
        # Preprocess
        input_blob = (image[0] * 255).astype(
            np.uint8
        )  # revert the image to original format and preprocess using the model

        # Forward
        embeddings = self.model.feature(input_blob)

        return embeddings


def loadModel():
    # URL, WEIGHTS_DIR_NAME, FILE_NAME은 download_model_and_get_filepath 메소드에서 지정할 수 있습니다.
    file_path = download_model_and_get_filepath()

    model = SFaceModel(model_path=file_path)
    # print('loading weight from ' + root_path + '/models/basemodels/weights/' + weight_file)

    return model

def download_model_and_get_filepath():
    URL="https://github.com/opencv/opencv_zoo/raw/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    WEIGHTS_DIR_NAME = 'weights'
    FILE_NAME = "face_recognition_sface_2021dec.onnx"

    # root_path = str(Path.cwd())
    
    weights_dir = os.path.join(script_dir, WEIGHTS_DIR_NAME)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    file_path = os.path.join(script_dir, weights_dir, FILE_NAME)
    if not os.path.isfile(file_path):
        print("sface model + weights will be downloaded...")
        gdown.download(URL, file_path, quiet=False)
    return file_path

def loadModelForSiameseNetwork():
    KERAS_MODEL_FILE_PATH = 'face_recognition_sface_2021dec_coverted.pth'

    onnx_model_path = download_model_and_get_filepath()
    file_path = os.path.join(os.path.dirname(onnx_model_path), KERAS_MODEL_FILE_PATH)
    model = ''
    if not os.path.isfile(file_path):
        print("sface model + weights will be converted...")

        # onnx-tf를 쓸 수가 없었음, 이건 구버전용. 대신 onnx2keras로 변경
        # onnx2keras는 버그가 많았고, 결정적으로 input shape 값이 달라져버리는 일이 발생하여 변환 실패 => onnx2torch를 이용해 torch 네트워크로 변환 후 keras로 변경 시도
        # ONNX 모델을 불러온 다음 Torch 모델로 변환합니다.
        # onnx_model = onnx.load(onnx_model_path)
        # torch_model = convert(onnx_model)
    else:
        # model = torch.load(file_path) # 파일로부터 모델을 불러옴
        pass

    return model
