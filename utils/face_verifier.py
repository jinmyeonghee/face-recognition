from models.basemodels.VGGFace import loadModel as vgg_load_model
from models.basemodels.Facenet512 import loadModel as facenet512_load_model
from models.basemodels.SFace import loadModel as sface_load_model

from .function.get_embedding import fetch_embedding
from .function.get_similarity import calculate_distance

class Verifier:
    def __init__(self, model_name = 'VGG-Face', distance_metric = 'cosine'):
        """
        ### models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]

        """
        self.model_name = model_name.lower()
        self.distance_metric = distance_metric
        if self.model_name == "VGG-FACE".lower() or model_name.lower() == "VGGFace".lower():
            self.model = vgg_load_model()
        elif self.model_name == "Facenet512".lower():
            self.model = facenet512_load_model()
        elif self.model_name == "SFace".lower():
            self.model = sface_load_model()

    def verify_each(self, origin_face, target_face):
        origin_embedding = fetch_embedding(self.model, origin_face)
        target_embedding = fetch_embedding(self.model, target_face)
        # 최종적으로 self.distance_metric을 사용해 get_distance 값을 가져온다.
        return calculate_distance(origin_embedding, target_embedding, self.model_name, self.distance_metric)

    
    def verify(self, origin_face_list, target_face_list, threshold= 0.5):
        """
        해당 메소드는 자체 threshold를 사용합니다.
        """
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
                similarity, result = self.verify_each(o_face, t_face)
                if result:
                    result_code = 2
                    result_message = '동일인이 존재합니다.'
                face_list.append(similarity)
            face_2dlist.append(face_list)
        
        return {'result_message': result_message, 'result_code': result_code, 'result_list': face_2dlist} 



    
        
        