import numpy as np
from models.basemodels.Gender import loadModel as gender_load_model

class GenderDistinguisher:

    def __init__(self):
        self.model = gender_load_model()

    def predict_gender(self, face_list:list):
        result_list = []
        img_list = np.array(face_list).astype('float32') / 255
        result_array = self.model.predict(img_list)
        for i, face_array in enumerate(result_array):
            gender_labels = ["Woman", "Man"]
            obj = {}
            obj["gender"] = {}
            for i, gender_label in enumerate(gender_labels):
                gender_prediction = round(100 * face_array[i], 2)
                obj["gender"][gender_label] = gender_prediction
            obj["dominant_gender"] = gender_labels[np.argmax(face_array)]

            result_list.append(obj)
        return result_list