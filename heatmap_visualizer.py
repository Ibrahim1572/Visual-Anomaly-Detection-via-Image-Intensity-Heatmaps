import numpy as np
import matplotlib.pyplot as plt
from anomaly_detector import detect_anomaly
# from anomaly_detector import get_processed_img
from image_loader import load_image

# path="sample_image.png"

# original_img=get_processed_img(path)
# anomaly_mask=detect_anomaly(original_img)
# amomaly_mask=anomaly_mask.astype(float)
# raw_img=load_image(path)

# def plot_heatmap():
#     fig = plt.figure(figsize=(12, 6))
#     gs = fig.add_gridspec(2, 3, height_ratios=[10, 1])

#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax3 = fig.add_subplot(gs[0, 2])

#     im1=ax1.imshow(raw_img, cmap='viridis', vmin=0, vmax=1)
#     ax1.set_title("Raw Image")

#     im2=ax2.imshow(original_img, cmap='viridis', vmin=0, vmax=1)
#     ax2.set_title("Color Distribution\n(based on Channel)")

#     im3=ax3.imshow(anomaly_mask, cmap='viridis', vmin=0, vmax=1)
#     ax3.set_title("Detected Anomalies")

#     cax = fig.add_subplot(gs[1, :])
#     fig.colorbar(im1, cax=cax, orientation='horizontal')
#     plt.tight_layout()
#     plt.show()

# def anomaly_overlay():
#     fig, axes=plt.subplots(1, 2)
#     im0=axes[1].imshow(original_img, cmap='viridis')
#     axes[1].imshow(anomaly_mask, cmap='Reds', alpha=0.5)
#     axes[0].imshow(raw_img, cmap='viridis')
#     plt.colorbar(im0, ax=axes)
#     plt.show()

def generate_heatmap(matrix:np.ndarray, colormap='viridis'):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap=colormap)
    fig.colorbar(im, ax=ax)
    return fig

def overlay_anomalies(matrix:np.ndarray, anomaly_mask):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='viridis')
    ax.imshow(anomaly_mask, cmap='Reds', alpha=0.5)
    fig.colorbar(im, ax=ax)
    return fig

# anomaly_overlay()
# plot_heatmap()