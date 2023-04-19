import os, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
sys.path.append(project_root)

from utils.face_detector import FacePreparer
from utils.face_verifier import Verifier # 특징을 각각 추출하여 함수로 비교
from utils.face_verifier2 import Verifier2 # 이미지를 둘다 넣고 딥러닝 결과값으로 비교 결과 확인
from utils.gender_distinguisher import GenderDistinguisher
from utils.function.generals import load_image

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
plt.rcParams['font.family'] = 'Malgun Gothic'

def is_numpy_image(array):
    return isinstance(array, np.ndarray) and (array.ndim == 3) and (array.shape[2] in [1, 3, 4])

class FaceDSProject:
    def __init__(self, min_detection_confidence = 0.2, model_name = 'VGG-Face', distance_metric = 'euclidean'):
        self.model_name = model_name
        self.preparer = FacePreparer(min_detection_confidence)
        # self.verifier = Verifier(self.model_name, distance_metric)
        self.verifier = Verifier2(self.model_name, distance_metric) #cosine, euclidean, euclidean_l2
        self.distinguisher = GenderDistinguisher()
    
    def get_faces(self, image_path, model_name='vggface'):
        """
        image_path : 이미지 url, 이미지 시스템 경로, 이미지 RGB np.ndarray 세 형식으로 받습니다.
        model 인풋사이즈에 맞게 전처리된 얼굴 이미지 numpy배열 리스트 추출 반환
        """
        image = load_image(image_path, project_root)
        if image is None or len(image) == 0:
            return None
       
        # np.ndarray에 어떤식으로 들어가는지 확인용.
        # with open(f"image_array{idx}.txt", "w") as outfile:
        #     for row in image:
        #         np.savetxt(outfile, row, fmt="%d", delimiter=",")
        return self.preparer.detect_faces(image, model_name)

    def verify(self, origin_image_path, target_image_path, threshold = 0.664):
        """
        verify한 결과 반환
        image_path : 이미지 url, 이미지 시스템 경로, 이미지 RGB np.ndarray 세 형식으로 받습니다.
        원본 이미지 얼굴별로 타겟 이미지 얼굴들과 비교 결과를 dict의 리스트로 반환.
        """
        face_list1 = self.get_faces(origin_image_path, self.model_name)
        face_list2 = self.get_faces(target_image_path, self.model_name)
        if face_list1 is None:
            return {'result_message' : '원본 이미지를 읽어올 수 없습니다.', 'result_code' : -22 }
        if face_list2 is None:
            return {'result_message' : '대상 이미지를 읽어올 수 없습니다.', 'result_code' : -21 }

        return self.verifier.verify(face_list1, face_list2, threshold)
    
    def distinguish(self, image_path):
        face_list = self.get_faces(image_path, 'gender')
        if face_list is None:
            return {'result_message' : '원본 이미지를 읽어올 수 없습니다.', 'result_code' : -11 }
        return self.distinguisher.predict_gender(face_list)

def plot_pairs(img1, img2, img3, img4, result):
    # 첫번째 이미지
    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.title('Origin')

    # 두번째 이미지
    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.title('Target')
    
    # 첫번째 이미지
    plt.subplot(2, 2, 3)
    plt.imshow(img3)
    plt.title('Origin_cropped')

    # 두번째 이미지
    plt.subplot(2, 2, 4)
    plt.imshow(img4)
    plt.title('Target_cropped')

    # subplot 간격 조절
    plt.subplots_adjust(hspace=0.4)

    plt.suptitle(result['result_message'])

    # 이미지 플롯 보여주기
    plt.show()
    # input('Press Enter to continue...')

def plot_genders(img1, img2, result1, img3, img4, result2):
    # 첫번째 이미지
    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.title((result1['result_list'][0]['dominant_gender']))

    # 두번째 이미지
    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.title((result1['result_list'][0]['dominant_gender'] + ' cropped'))
    
    # 첫번째 이미지
    plt.subplot(2, 2, 3)
    plt.imshow(img3)
    plt.title((result2['result_list'][0]['dominant_gender']))

    # 두번째 이미지
    plt.subplot(2, 2, 4)
    plt.imshow(img4)
    plt.title((result2['result_list'][0]['dominant_gender'] + ' cropped'))

    # subplot 간격 조절
    plt.subplots_adjust(hspace=0.4)

    # 이미지 플롯 보여주기
    plt.show()
    
if __name__ == '__main__':
    # min_detection_confidence => detecting 임계값(0 ~ 1)
    # model_name => vggface/vgg-face, facenet512 (모델은 대소문자 구분 없음)
    # distance_metric => cosine, euclidean, euclidean_l2
    project = FaceDSProject(model_name='vggface', distance_metric='euclidean')

    source1 = '../datasets/High_Resolution/19062421/S001/L1/E01/C6.jpg'
    source2 = '../datasets/High_Resolution/19062421/S001/L1/E01/C7.jpg'
    source3 = '../datasets/_temp/base/201703240905286710_1.jpg'
    
    img00 = 'sample_data/s3.jpg'
    img01 = 'sample_data/s3_2.jpg'
    img10 = 'sample_data/s2.jpg'

    img_m = 'sample_data/s8.jpg'
    img_w = 'sample_data/s1.jpg'

    print('This is sample')

    result1 = project.verify(img00, img01)
    print('결과 메세지 : ', result1['result_message'], ' 결과 값 : ', result1['result_code'])
    print('유사도 : ', result1['result_list'])
    plot_pairs(load_image(img00, project_root), load_image(img01, project_root), project.get_faces(img00)[0], project.get_faces(img01)[0], result1)
    
    result2 = project.verify(img00, img10)
    print('결과 메세지 : ', result2['result_message'], ' 결과 값 : ', result2['result_code'])
    print('유사도 : ', result2['result_list'])
    plot_pairs(load_image(img00, project_root), load_image(img10, project_root), project.get_faces(img00)[0], project.get_faces(img10)[0], result2)
    
    result_m = project.distinguish(img_m)
    result_w = project.distinguish(img_w)
    plot_genders(load_image(img_m, project_root), project.get_faces(img_m)[0], result_m, load_image(img_w, project_root), project.get_faces(img_w)[0], result_w)

    # url1 = 'https://m.media-amazon.com/images/I/71ZMw9YqEJL._SL1500_.jpg'
    # url2 = 'https://m.media-amazon.com/images/I/71bnIcDHk6L._SL1500_.jpg'
    
    # print(project.verify(url1, url1))