import numpy as np
import urllib.request
from PIL import Image
from io import BytesIO

def url_to_np_array(url):
    """
    url로부터 RGB 이미지 배열을 반환합니다.
    """
    with urllib.request.urlopen(url) as response:
        image_data = response.read()

    image = Image.open(BytesIO(image_data))
    rgb_image_array = np.array(image)
    
    return rgb_image_array