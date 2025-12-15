from PIL import Image
import numpy as np

def load_image(path="sample_image.png"):
    img=Image.open(path)
    arr=np.array(img)
    arr=arr[:, :, :3]
    return arr
