import os
from pathlib import Path
from models.basemodels import Facenet


def loadModel():

    model = Facenet.InceptionResNetV2(dimension=512)
    
    root_path = str(Path.cwd())
    weight_file = "facenet512_weights.h5"
    model.load_weights(root_path + '/models/basemodels/weights/' + weight_file)
    print('loading weight from ' + root_path + '/models/basemodels/weights/' + weight_file)

    return model
