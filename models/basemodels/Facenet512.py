import os
from pathlib import Path
from models.basemodels import Facenet

script_dir = os.path.dirname(os.path.abspath(__file__))

def loadModel():

    model = Facenet.InceptionResNetV2(dimension=512)
    
    root_path = str(Path.cwd())
    weight_file = "facenet512_weights.h5"
    model.load_weights(os.path.join(script_dir, 'weights', weight_file))
    # print('loading weight from ' + root_path + '/models/basemodels/weights/' + weight_file)

    return model
