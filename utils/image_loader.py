import cv2, os
import numpy as np
from .function.url_to_image import url_to_np_array

def imread_korean(file_path):
    img_array = np.fromfile(file_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def image_loader(imagelike, project_root):
    """
    이미지 경로로부터 RGB 이미지 배열을 추출하여 반환합니다.
    mediapipe가 RGB 이미지 배열을 사용하기 때문에 RGB로 추출합니다.
    """
    if imagelike.lower().startswith("http://") or imagelike.lower().startswith("https://"):
        return url_to_np_array(imagelike)
    else:
        image_realpath = os.path.join(project_root, imagelike)
        img = cv2.imread(image_realpath)
        if img is None:
            img = imread_korean(image_realpath)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb_img