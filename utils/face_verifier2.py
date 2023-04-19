from models.basemodels.CS_AI16_VGGFace import loadSiameseModel as vgg_load_siamese_model
from models.basemodels.CS_AI16_Facenet512 import loadSiameseModel as facenet512_load_siamese_model
from models.basemodels.CS_AI16_SFace import loadSiameseModel as sface_load_siamese_model
from models.basemodels.CS_AI16_ArcFace import loadSiameseModel as arcface_load_siamese_model
import tensorflow as tf


class Verifier2:
    def __init__(self, model_name = 'VGG-Face', distance_metric = 'cosine'):
        """
        ### models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]

        """
        self.model_name = model_name.lower()
        self.distance_metric = distance_metric
        if self.model_name == "VGG-FACE".lower() or model_name.lower() == "VGGFace".lower():
            self.model = vgg_load_siamese_model(distance_metric)

        elif self.model_name == "Facenet512".lower():
            self.model = facenet512_load_siamese_model(distance_metric)
            
        elif self.model_name == "SFace".lower():
            self.model = sface_load_siamese_model(distance_metric)
        
        elif self.model_name == "ArcFace".lower():
            self.model = arcface_load_siamese_model(distance_metric)

    def verify_each(self, origin_face, target_face):
        origin_face_batch = tf.expand_dims(origin_face / 255, axis=0)
        target_face_batch = tf.expand_dims(target_face / 255, axis=0)
        return self.model.predict([origin_face_batch, target_face_batch])


    
    def verify(self, origin_face_list, target_face_list, threshold = 0.664):
        
        if len(origin_face_list) == 0:
            
            return {'result_message' : '원본 이미지에서 얼굴이 검출되지 않았습니다.', 'result_code' : -2 }
        if len(target_face_list) == 0:
            return {'result_message' : '비교할 이미지에서 얼굴이 검출되지 않았습니다.', 'result_code' : -1 }
        
        # 각각 verify_each 함수를 돌린 결과값을 result로 뽑고 list모양인 dict값에 append한다.
        face_2dlist = []
        result_code = 0
        result_message = '동일인이 존재하지 않습니다.'
        for i, o_face in enumerate(origin_face_list):
            face_list = []
            for j, t_face in enumerate(target_face_list):
                result = self.verify_each(o_face, t_face)
                if result[0][0] > threshold:
                    result_code = 2
                    result_message = '동일인이 존재합니다.'
                face_list.append(result[0][0])
            face_2dlist.append(face_list)
        
        return {'result_message': result_message, 'result_code': result_code, 'result_list': face_2dlist, 'origin' : origin_face_list, 'target' : target_face_list} 



    
        
        