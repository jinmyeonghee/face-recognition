import os, gdown
from pathlib import Path
from models.basemodels import Facenet

script_dir = os.path.dirname(os.path.abspath(__file__))

def loadModel():
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5"
    weights_dir_name = 'weights'
    weight_file = "facenet512_weights.h5"

    model = Facenet.InceptionResNetV2(dimension=512)
    
    # root_path = str(Path.cwd())

    weights_dir = os.path.join(script_dir, weights_dir_name)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    file_path = os.path.join(script_dir, weights_dir_name, weight_file)
    if os.path.isfile(file_path) != True:
        print("facenet512_weights.h5 will be downloaded...")
        gdown.download(url, file_path, quiet=False)
    model.load_weights(file_path)

    return model
