from image_loader import load_image
import numpy as np
import matplotlib.pyplot as plt

img_arr=load_image()

def normalizer(img_arr: np.ndarray):
    img_arr=img_arr.astype(np.float64)
    
    normalized_img=(img_arr-img_arr.min())/(img_arr.max()-img_arr.min())
    return normalized_img


def get_greyscale_img(img_arr: np.ndarray):
    grey_scale_img=img_arr[:, :, 0]*0.2898+img_arr[:, :, 1]*0.5870+img_arr[:, :, 2]*0.1140
    grey_scale_img=normalizer(grey_scale_img)
    return grey_scale_img
    
def get_rgb_avg_img(img_arr: np.ndarray):
    rgb_avg_img=(img_arr[:, :, 0]+img_arr[:, :, 1]+img_arr[:, :, 2])/3
    rgb_avg_img=normalizer(rgb_avg_img)
    return rgb_avg_img

def get_channel_img(img_arr: np.ndarray, channel: str):
    if(channel=='r' or channel=='R'):
        channel_img=img_arr[:, :, 0]
        
    elif(channel=='G' or channel=='g'):
        channel_img=img_arr[:, :, 1]
    
    elif(channel=='B' or channel=='b'):
        channel_img=img_arr[:, :, 2]
        
    else:
        raise ValueError("Invalid channel")
    
    channel_img=normalizer(channel_img)    
    return channel_img


# norm=normalizer(img_arr)
# print(type(norm[1, 1, 1]))
# print(type(img_arr[1 ,1,1]))
# gray = get_greyscale_img(img_arr)
# print(gray.shape, gray.dtype)
# plt.imshow(gray, cmap='gray'); plt.show()   

# rgb_avg = get_rgb_avg_img(img_arr)
# print(rgb_avg.shape, rgb_avg.dtype)
# plt.imshow(rgb_avg, cmap='gray')
# plt.show()
 
# r = get_channel_img(img_arr, 'B')
# print(r.shape, r.dtype)
# plt.imshow(r, cmap='gray')
# plt.show()


# print(get_greyscale_img(img_arr).shape)
# print(get_channel_img(img_arr, "B").shape)
# print(get_rgb_avg_img(img_arr).shape)
# print(normalizer(img_arr).shape)
# print(img_arr.shape)