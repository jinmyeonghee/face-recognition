import tensorflow as tf
from models.basemodels.VGGFace import loadModel as vgg_load_model
from models.basemodels.Facenet512 import loadModel as facenet512_load_model
from models.basemodels.SFace import loadModel as sface_load_model

from .function.get_embedding import get_embedding
from .function.get_similarity import get_distance

class verifier:
    def __init__(self, model = 'VGG-Face', distance_metric = 'cosine'):
        """
        ### models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
        Verifier 안에서는 매개변수 model이 model_name Str변수가 되고, 실제 모델이 model 변수에 할당된다.

        """
        self.model_name = model.capitalize()
        self.distance_metric = distance_metric
        if self.model_name == "VGG-FACE".capitalize() or model.capitalize() == "VGGFace".capitalize():
            self.model = vgg_load_model()
        elif self.model_name == "Facenet512".capitalize():
            self.model = facenet512_load_model()
        elif self.model_name == "SFace".capitalize():
            self.model = sface_load_model()

    def verify_each(self, origin_face, target_face):
        origin_embedding = get_embedding(self.model, origin_face)
        target_embedding = get_embedding(self.model, target_face)
        # 최종적으로 self.distance_metric을 사용해 get_distance 값을 가져온다.
        return get_distance(origin_embedding, target_embedding, self.model_name, self.distance_metric)

    
    def verify(self, origin_face_list, target_face_list):
        print(origin_face_list)
        print(len(origin_face_list))
        print(target_face_list)
        print(len(target_face_list))
        if len(origin_face_list) == 0:
            
            return {'result_message' : '원본 이미지에서 얼굴이 검출되지 않았습니다.', 'result_code' : -2 }
        if len(target_face_list) == 0:
            return {'result_message' : '비교할 이미지에서 얼굴이 검출되지 않았습니다.', 'result_code' : -1 }
        
        # 각각 verify_each 함수를 돌린 결과값을 result로 뽑고 list모양인 dict값에 append한다.
        face_dict_list = []
        for i, o_face in enumerate(origin_face_list):
            face_dict = {f"{i+1}번째 얼굴" : []}
            for j, t_face in enumerate(target_face_list):
                result = self.verify_each(o_face, t_face)
                face_dict[f"{i+1}번째 얼굴"].append(result)
            face_dict_list.append(face_dict)
        
        return face_dict_list



    
        
        