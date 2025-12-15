import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

from image_loader import load_image
from intensity_processor import (
    get_rgb_avg_img,
    get_greyscale_img,
    get_channel_R_img,
    get_channel_G_img,
    get_channel_B_img
)
from anomaly_detector import compute_stats, detect_anomaly
from heatmap_visualizer import generate_heatmap, overlay_anomalies
from stats_analyzer import compute_global_stats

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Visual Anomaly Detection via Image Intensity Heatmaps",
    layout="wide",
)

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "view_mode" not in st.session_state:
    st.session_state.view_mode = False

# --------------------------------------------------
# Colors
# --------------------------------------------------
LIGHT_BG = "#DBE2DC"
DARK_BG = "#0F172A"
PRIMARY = "#335765"
CARD_BG = "#B6D9E0"

bg_color = DARK_BG if st.session_state.dark_mode else LIGHT_BG
text_color = "#E5E7EB" if st.session_state.dark_mode else PRIMARY
card_color = "#1E293B" if st.session_state.dark_mode else CARD_BG

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {bg_color}; }}
    h1, h2, h3, h4, h5 {{ color: {text_color}; }}
    .stat-card {{
        background-color: {card_color};
        padding: 1.2rem;
        border-radius: 14px;
        text-align: center;
        font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>Visual Anomaly Detection via Image Intensity Heatmaps</h1>",
    unsafe_allow_html=True,
)
st.divider()

# --------------------------------------------------
# Controls Below Title
# --------------------------------------------------
col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 1, 1])

# Dark Mode Toggle
with col1:
    # st.session_state.dark_mode = st.checkbox("Toggle Mode\n☀️Light/Dark🌙", value=st.session_state.dark_mode)
    if st.button(" Toggle Mode\n☀️Light/Dark🌙"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        # Streamlit automatically re-runs

# Image Source Dropdown
with col2:
    image_source = st.selectbox(
        "Image Source",
        ["Upload Image", "Camera Capture", "Sample Image"]
    )

# Intensity Mode Dropdown
with col3:
    intensity_mode = st.selectbox(
        "Intensity Mode",
        ["Grayscale", "RGB Average", "Red", "Green", "Blue"],
    )

# Anomaly Threshold Slider
with col4:
    k = st.slider("Anomaly Threshold (k)", 0.5, 5.0, 2.0, 0.1)

# Colormap Dropdown
with col5:
    colormap = st.selectbox(
        "Colormap",
        ["viridis", "plasma", "inferno", "magma", "cividis"],
    )

# --------------------------------------------------
# Load Image
# --------------------------------------------------
uploaded_image = None

if image_source == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
elif image_source == "Camera Capture":
    uploaded_image = st.camera_input("Capture Image")
elif image_source == "Sample Image":
    sample_path = st.selectbox(
        "Select Sample",
        [
            r"D:\AI_Lab_Project_Sem3\Visual-Anomaly-Detection-via-Image-Intensity-Heatmaps\sample_image.png",
            r"D:\AI_Lab_Project_Sem3\Visual-Anomaly-Detection-via-Image-Intensity-Heatmaps\Hitler.png",
        ]
    )
    uploaded_image = sample_path

if uploaded_image is None:
    st.warning("Please select or upload an image to begin.")
    st.stop()

if isinstance(uploaded_image, str):
    original_image = load_image(uploaded_image)
else:
    image = Image.open(uploaded_image)
    original_image = np.array(image)

# --------------------------------------------------
# Intensity Processing
# --------------------------------------------------
if intensity_mode == "Grayscale":
    intensity = get_greyscale_img(original_image)
elif intensity_mode == "RGB Average":
    intensity = get_rgb_avg_img(original_image)
elif intensity_mode == "Red":
    intensity = get_channel_R_img(original_image)
elif intensity_mode == "Green":
    intensity = get_channel_G_img(original_image)
elif intensity_mode == "Blue":
    intensity = get_channel_B_img(original_image)

# --------------------------------------------------
# Anomaly Detection
# --------------------------------------------------
anomaly_mask = detect_anomaly(intensity, k)
stats = compute_global_stats(intensity, anomaly_mask)

# --------------------------------------------------
# Generate Heatmap and Overlay Figures (for display & download)
# --------------------------------------------------
fig_heatmap = generate_heatmap(intensity, colormap)
fig_overlay = overlay_anomalies(intensity, anomaly_mask)

# --------------------------------------------------
# View Mode Button
# --------------------------------------------------
if st.button("🔄 Switch View Results Mode (Visual/Statistical)"):
    st.session_state.view_mode = not st.session_state.view_mode

# --------------------------------------------------
# Visual Results View
# --------------------------------------------------
if not st.session_state.view_mode:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(original_image, width=350)

    with col2:
        st.subheader("Heatmap")
        st.pyplot(fig_heatmap)

    with col3:
        st.subheader("Overlay")
        st.pyplot(fig_overlay)

# --------------------------------------------------
# Statistical Results View
# --------------------------------------------------
else:
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.markdown(f"<div class='stat-card'>Mean<br>{stats['mean']:.3f}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='stat-card'>Min<br>{stats['min']:.3f}</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='stat-card'>Max<br>{stats['max']:.3f}</div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='stat-card'>% Anomalies<br>{stats['percent_anomalies']:.2f}%</div>", unsafe_allow_html=True)
    c5.markdown(f"<div class='stat-card'>Total Anomalies<br>{anomaly_mask.sum():.0f}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Detailed Statistics"):
        st.json(stats)

# --------------------------------------------------
# Download Section
# --------------------------------------------------
st.divider()
st.subheader("Downloads")

# Heatmap download
buf_heatmap = io.BytesIO()
fig_heatmap.savefig(buf_heatmap, format="png")
st.download_button("⬇️ Heatmap", buf_heatmap.getvalue(), "heatmap.png")

# Anomaly mask download
buf_mask = io.BytesIO()
plt.imsave(buf_mask, anomaly_mask, cmap="gray")
st.download_button("⬇️ Anomaly Mask", buf_mask.getvalue(), "anomaly_mask.png")
