import numpy as np
import cv2
from PIL import Image

def pil2cv2(pil_img):
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def cv2topil(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR))

