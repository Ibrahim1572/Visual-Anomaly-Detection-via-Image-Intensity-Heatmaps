import numpy as np
import matplotlib.pyplot as plt
from intensity_processor import get_greyscale_img 
from intensity_processor import get_rgb_avg_img
from intensity_processor import get_channel_img
from image_loader import load_image

def compute_stats(processed_img: np.ndarray):
    mean_img=processed_img.mean()
    min_img=processed_img.min()
    max_img=processed_img.max()
    std_img=processed_img.std()
    
    stats=[mean_img, min_img, max_img, std_img]
    return stats

def detect_anomaly(processed_img:np.ndarray, k=1.5):
    stats=compute_stats(processed_img)
    mean_val=stats[0]
    min_val=stats[1]
    max_val=stats[2]
    std_val=stats[3]
    
    anomaly_mask = np.abs(processed_img - mean_val) > (k * std_val)
        
    return anomaly_mask

# def get_processed_img(path: str):
#     img_arr=load_image(path)
#     img_arr=img_arr.astype(np.float64)
#     while(True):
#         ch1=int(input("Enter option:\n1. Greyscale\n2. RGB Avg\n3. Single Channel\nYour Choice: "))
#         if ch1==1:
#             process_img=get_greyscale_img(img_arr)
#             break
#         elif ch1==2:
#             process_img=get_rgb_avg_img(img_arr)
#             break
#         elif ch1==3:
#             while(True):
#                 ch2=str(input("Enter option (R, G, B): "))
#                 ch2=ch2.lower()
#                 if ch2=="r":
#                     process_img=get_channel_img(img_arr ,"R")
#                     break
#                 elif ch2=='g':
#                     process_img=get_channel_img(img_arr ,"G")
#                     break
#                 elif ch2=='b':
#                     process_img=get_channel_img(img_arr ,"B")
#                     break
#             break
#     return process_img

# print(detect_anaomaly(process_img).shape)   
# print(process_img.shape)              
# print(detect_anaomaly(process_img, 2))

# gray = detect_anomaly(get_processed_img(), 1.5)
# print(gray.shape, gray.dtype)
# plt.imshow(gray, cmap='gray'); plt.show()
