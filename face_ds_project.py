import os
import sys

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_root)

from utils.face_detector import representer
from utils.face_verifier import verifier





class FaceDSProject:
    def __init__(self, min_detection_confidence = 0.2, model = 'VGG-Face', distance_metric = 'cosine'):
        self.representer = representer(min_detection_confidence=0.2)
        self.verifier = verifier(model, distance_metric)
    
    def represent(self, image_path):
        """
        224 X 224 전처리된 얼굴 이미지 numpy배열 리스트 추출 반환
        """
        return self.representer.detect_faces(image_path)

    def verify(self, origin_image_path, target_image_path):
        """
        verify한 결과 반환
        """
        face_list1 = self.represent(origin_image_path)
        face_list2 = self.represent(target_image_path)

        return self.verifier.verify(face_list1, face_list2)

if __name__ == '__main__':

    project = FaceDSProject()

    source1 = '../datasets/High_Resolution/19062421/S001/L1/E01/C6.jpg'
    source2 = '../datasets/High_Resolution/19062421/S001/L1/E01/C7.jpg'
    
    print('This is sample')
    print(project.verify(source1, source2))