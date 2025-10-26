import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
import cv2
import os

# ==============================
# LOAD MODEL
# ==============================

# Load klasifikasi .h5
classifier_model_path = "model/elfi_Laporan_2.h5"
classifier_model = None
if os.path.exists(classifier_model_path):
    classifier_model = tf.keras.models.load_model(classifier_model_path)

# Load YOLO .pt
yolo_model_path = "model/Elfii_Laporan_4.pt"
yolo_model = None
if os.path.exists(yolo_model_path):
    yolo_model = YOLO(yolo_model_path)

# ==============================
# UI Styling
# ==============================
st.set_page_config(page_title="Vision Inspector Dashboard", page_icon="🚗", layout="wide")

st.markdown("""
    <style>
        .title {
            text-align: left;
            font-size: 34px;
            font-weight: bold;
            color: white;
            margin-bottom: -15px;
        }
        .subtitle {
            text-align: left;
            font-size: 18px;
            color: #ffe7ee;
        }
        .header-box {
            background: linear-gradient(90deg,#e04f80,#ff87a5);
            padding: 25px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1,5])

with col2:
    st.markdown('<div class="header-box">', unsafe_allow_html=True)
    st.markdown('<p class="title">🚗 Vision Inspector Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Analisis Kendaraan & Jajanan Tradisional</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("⚙ Panel Kontrol")
mode = st.sidebar.selectbox("Mode Analisis:", ["Klasifikasi", "Deteksi (YOLO)"])

# ==============================
# UPLOAD IMAGE
# ==============================
uploaded = st.file_uploader("Unggah gambar:", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Gambar Terunggah", use_column_width=True)

    # Convert numpy
    img_np = np.array(img)

    # ==============================
    # KLASIFIKASI
    # ==============================
    if mode == "Klasifikasi":
        if classifier_model:
            resized = cv2.resize(img_np, (224,224))
            normalized = resized.reshape(1,224,224,3) / 255.0
            preds = classifier_model.predict(normalized)

            labels = ["Bike","Car","Motorcycle","Plane","Ship",
                      "Semprong","Grontol","Lanting","Lumpia","Putu Ayu","Wajik"]

            pred_idx = np.argmax(preds)
            st.success(f"✅ Prediksi: **{labels[pred_idx]}**")
        else:
            st.error("❌ Model klasifikasi tidak ditemukan!")

    # ==============================
    # DETEKSI YOLO
    # ==============================
    if mode == "Deteksi (YOLO)":
        if yolo_model:
            results = yolo_model(img_np)
            result_img = results[0].plot()
            st.image(result_img, caption="Hasil Deteksi", use_column_width=True)
        else:
            st.error("❌ Model YOLO tidak ditemukan!")
else:
    st.warning("Silahkan upload gambar terlebih dahulu.")
