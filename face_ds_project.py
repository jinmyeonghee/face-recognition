import os, sys
import numpy as np
import tensorflow as tf

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
sys.path.append(project_root)

from utils.face_detector import FacePreparer
from utils.face_verifier import Verifier # 특징을 각각 추출하여 함수로 비교
from utils.face_verifier2 import Verifier2 # 이미지를 둘다 넣고 딥러닝 결과값으로 비교 결과 확인
from utils.gender_distinguisher import GenderDistinguisher
from utils.function.generals import load_image

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def is_numpy_image(array):
    return isinstance(array, np.ndarray) and (array.ndim == 3) and (array.shape[2] in [1, 3, 4])

class FaceDSProject:
    def __init__(self, min_detection_confidence = 0.2, model_name = 'VGG-Face', distance_metric = 'cosine'):
        self.model_name = model_name
        self.preparer = FacePreparer(min_detection_confidence)
        # self.verifier = Verifier(self.model_name, distance_metric)
        self.verifier = Verifier2(self.model_name, distance_metric)
        self.distinguisher = GenderDistinguisher()
    
    def get_faces(self, image_path, model_name='vggface'):
        """
        image_path : 이미지 url, 이미지 시스템 경로, 이미지 RGB np.ndarray 세 형식으로 받습니다.
        model 인풋사이즈에 맞게 전처리된 얼굴 이미지 numpy배열 리스트 추출 반환
        """
        image = load_image(image_path, project_root)
       
        # np.ndarray에 어떤식으로 들어가는지 확인용.
        # with open(f"image_array{idx}.txt", "w") as outfile:
        #     for row in image:
        #         np.savetxt(outfile, row, fmt="%d", delimiter=",")
        return self.preparer.detect_faces(image, model_name)

    def verify(self, origin_image_path, target_image_path):
        """
        verify한 결과 반환
        image_path : 이미지 url, 이미지 시스템 경로, 이미지 RGB np.ndarray 세 형식으로 받습니다.
        원본 이미지 얼굴별로 타겟 이미지 얼굴들과 비교 결과를 dict의 리스트로 반환.
        """
        face_list1 = self.get_faces(origin_image_path, self.model_name)
        face_list2 = self.get_faces(target_image_path, self.model_name)

        return self.verifier.verify(face_list1, face_list2)
    
    def distinguish(self, image_path):
        face_list = self.get_faces(image_path, 'gender')
        return self.distinguisher.predict_gender(face_list)

    
if __name__ == '__main__':
    # min_detection_confidence => detecting 임계값(0 ~ 1)
    # model_name => vggface/vgg-face, facenet512, sface (모델은 대소문자 구분 없음)
    # distance_metric => cosine, euclidean, euclidean_l2
    project = FaceDSProject(model_name='vggface', distance_metric='cosine')

    source1 = '../datasets/High_Resolution/19062421/S001/L1/E01/C6.jpg'
    source2 = '../datasets/High_Resolution/19062421/S001/L1/E01/C7.jpg'
    source3 = '../datasets/_temp/base/201703240905286710_1.jpg'
    
    print('This is sample')
    print(project.verify(source1, source1))
    # print(project.distinguish(source1))
    # print(project.distinguish(source2))
    # print(project.distinguish(source3))

    url1 = 'https://m.media-amazon.com/images/I/71ZMw9YqEJL._SL1500_.jpg'
    url2 = 'https://m.media-amazon.com/images/I/71bnIcDHk6L._SL1500_.jpg'
    # url2 = '../datasets/_temp/base/bts01.jpg'
    
    print(project.verify(url1, url1))