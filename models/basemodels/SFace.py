import os, gdown
import numpy as np
from pathlib import Path
import cv2 as cv

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
    url="https://github.com/opencv/opencv_zoo/raw/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    weights_dir_name = 'weights'
    file_name = "face_recognition_sface_2021dec.onnx"

    # root_path = str(Path.cwd())
    
    weights_dir = os.path.join(script_dir, weights_dir_name)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    file_path = os.path.join(script_dir, weights_dir, file_name)
    if not os.path.isfile(file_path):
        print("sface model + weights will be downloaded...")
        gdown.download(url, file_path, quiet=False)

    model = SFaceModel(model_path=file_path)
    # print('loading weight from ' + root_path + '/models/basemodels/weights/' + weight_file)

    return model
