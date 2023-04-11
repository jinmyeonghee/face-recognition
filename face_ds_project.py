import sys
import os

root_path = os.path.abspath('.') # 현재 작업 디렉토리를 루트로 가정합니다.
sys.path.insert(0, root_path) # 루트 디렉토리를 sys.path에 추가합니다.

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

    source1 = 'https://images.sbs.com.au/dims4/default/8611bed/2147483647/strip/true/crop/1920x1080+0+0/resize/1280x720!/quality/90/?url=http%3A%2F%2Fsbs-au-brightspot.s3.amazonaws.com%2Fdrupal%2Ftopics%2Fpublic%2Fimages%2Ftwg_articles%2Ffotojet_-_2020-01-21t140318.009.jpg'
    source2 = 'https://cdn.britannica.com/78/128778-050-3BEDC27B/Zinedine-Zidane-ball-final-World-Cup-Italy-2006.jpg'
    
    print('This is sample')
    print(project.verify(source1, source2))