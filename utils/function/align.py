import math
import numpy as np
from PIL import Image

def findEuclideanDistance(source_representation, target_representation):
    """
    주어진 두 벡터의 유클리드 거리를 계산하는 함수
    
    Args:
    - source_representation : 1차원 numpy 배열. 벡터의 차원은 같아야 합니다.
    - test_representation : 1차원 numpy 배열. 벡터의 차원은 같아야 합니다.
    
    Returns:
    - distance : 주어진 두 벡터의 유클리드 거리
    """
    
    # 벡터 차원이 다른 경우 에러 메시지 출력
    if source_representation.shape != target_representation.shape:
        raise ValueError("두 벡터의 차원이 다릅니다.")
    
    # 각 원소의 차이를 계산한 후 제곱합니다.
    diff = source_representation - target_representation
    squared_diff = diff**2
    
    # 각 차이들의 합을 계산한 후 제곱근을 취해 거리를 계산합니다.
    distance = np.sqrt(np.sum(squared_diff))
    
    return distance

def alignment_procedure(img, right_eye, left_eye):

    # this function aligns given face in img based on left and right eye coordinates

    right_eye_x, right_eye_y = right_eye
    left_eye_x, left_eye_y = left_eye

    # -----------------------
    # find rotation direction

    if right_eye_y > left_eye_y:
        point_3rd = (left_eye_x, right_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (right_eye_x, left_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(left_eye), np.array(right_eye))

    # -----------------------

    # apply cosine rule

    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

    # -----------------------

    return img  # return img anyway