import numpy as np
from anomaly_detector import detect_anomaly
# from anomaly_detector import get_processed_img
from image_loader import load_image


# original_img=get_processed_img(path)
# anomaly_mask=detect_anomaly(original_img)
# amomaly_mask=anomaly_mask.astype(float)
# raw_img=load_image(path)

def compute_global_stats(original_img: np.ndarray, anomaly_mask: np.ndarray):
    meann=original_img.mean()
    minn=original_img.min()
    maxx=original_img.max()
    
    tss=anomaly_mask.size
    summ=anomaly_mask.sum()
    percentagee=(summ/tss)*100
    
    # return [meann, minn, maxx, percentagee]
    return {
        "mean": original_img.mean(),
        "min": original_img.min(),
        "max": original_img.max(),
        "percent_anomalies": (anomaly_mask.sum() / anomaly_mask.size) * 100,
        "total_anomalies": summ
    }

# print(compute_global_stats(original_img, anomaly_mask))