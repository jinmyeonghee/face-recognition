import onnx
from onnx_tf.backend import prepare

def load_onnx_model(model_path):
    # ONNX 모델을 불러옵니다.
    onnx_model = onnx.load(model_path)

    # ONNX 모델을 TensorFlow 형식으로 변환합니다.
    return prepare(onnx_model)