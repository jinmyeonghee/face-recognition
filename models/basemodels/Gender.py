import os
import gdown
import tensorflow as tf
from . import VGGFace

# -------------------------------------
# pylint: disable=line-too-long
# -------------------------------------
# dependency configurations
script_dir = os.path.dirname(os.path.abspath(__file__))

tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import Convolution2D, Flatten, Activation
elif tf_version == 2:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation
# -------------------------------------

# Labels for the genders that can be detected by the model.
labels = ["Woman", "Man"]


def loadModel(
    url="https://dl.dropboxusercontent.com/s/j99xq4vgyt4xt09/gender_train_weights7.h5",     # 새로 학습시킨 가중치
):
    weights_dir_name = 'weights'
    file_name = "gender_train_weights7.h5"
    base_file_name = "gender_model_weights.h5"
    
    url_base="https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5"    # 기존 가중치
    
    model = VGGFace.baseModel()

    # --------------------------

    classes = 2
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name="predictions")(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation("softmax")(base_model_output)

    # --------------------------

    gender_model = Model(inputs=model.input, outputs=base_model_output)

    # --------------------------

    # load weights
    weights_dir = os.path.join(script_dir, weights_dir_name)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    file_path = os.path.join(script_dir, weights_dir, file_name)
    base_file_path = os.path.join(script_dir, weights_dir, base_file_name)

    if os.path.isfile(file_path) != True:
        print("gender_train_weights7.h5 will be downloaded...")
        try:
            gdown.download(url, file_path, quiet=False) 
        except:
            print("failed to download gender_train_weights7.h5")
            print("gender_model_weights.h5 will be downloaded...")
            gdown.download(url_base, base_file_path, quiet=False) 

    gender_model.load_weights(file_path)

    return gender_model
