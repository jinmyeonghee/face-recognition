import numpy as np
import urllib.request
from PIL import Image
from io import BytesIO

def url_to_np_array(url):
    with urllib.request.urlopen(url) as response:
        image_data = response.read()

    image = Image.open(BytesIO(image_data))
    image_array = np.array(image)
    
    return image_array