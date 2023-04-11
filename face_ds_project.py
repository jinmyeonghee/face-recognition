import os, sys
import numpy as np

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
sys.path.append(project_root)

from utils.face_detector import representer
from utils.face_verifier import verifier
from utils.image_loader import image_loader


def is_numpy_image(array):
    return isinstance(array, np.ndarray) and (array.ndim == 3) and (array.shape[2] in [1, 3, 4])

class FaceDSProject:
    def __init__(self, min_detection_confidence = 0.2, model = 'VGG-Face', distance_metric = 'cosine'):
        self.representer = representer(min_detection_confidence=0.2)
        self.verifier = verifier(model, distance_metric)
    
    def represent(self, image_path):
        """
        image_path : 이미지 url, 이미지 시스템 경로, 이미지 RGB np.ndarray 세 형식으로 받습니다.
        224 X 224 전처리된 얼굴 이미지 numpy배열 리스트 추출 반환
        """
        if isinstance(image_path, str) == True:
            image = image_loader(image_path, project_root)
        elif is_numpy_image(image_path) == True:
            image = image_path
        else:    
            raise ValueError('이미지 경로가 올바른 url이나 시스템 경로를 의미하는 str가 아니거나, 이미지 np.ndarray가 아닙니다.')
        
        # np.ndarray에 어떤식으로 들어가는지 확인용.
        # with open(f"image_array{idx}.txt", "w") as outfile:
        #     for row in image:
        #         np.savetxt(outfile, row, fmt="%d", delimiter=",")
        return self.representer.detect_faces(image)

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


    url1 = 'https://m.media-amazon.com/images/I/71ZMw9YqEJL._SL1500_.jpg'
    # url1 = '../datasets/_temp/base/bts01.jpg'
    url2 = 'https://m.media-amazon.com/images/I/71bnIcDHk6L._SL1500_.jpg'
    # url2 = '../datasets/_temp/base/bts01.jpg'
    
    print(project.verify(url1, url2))