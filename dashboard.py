import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import tensorflow as tf
from ultralytics import YOLO

# =============================================
# LOAD MODEL KLASIFIKASI
# =============================================

clf_path = "model_klasifikasi.h5"

try:
    classifier = tf.keras.models.load_model(clf_path)
    clf_status = True
except:
    classifier = None
    clf_status = False

# =============================================
# LOAD MODEL YOLO
# =============================================

yolo_path = "yolo.pt"

try:
    yolo_model = YOLO(yolo_path)
    yolo_status = True
except:
    yolo_model = None
    yolo_status = False

# =============================================
# STREAMLIT UI
# =============================================

st.sidebar.title("⚙ Panel Kontrol")

mode = st.sidebar.selectbox("Mode Analisis:", ["Klasifikasi", "Deteksi (YOLO)"])

uploaded_image = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption="Gambar Input", width=500)

# =============================================
# MODE KLASIFIKASI
# =============================================

if mode == "Klasifikasi":

    if not clf_status:
        st.error("❌ Classifier tidak ditemukan!")
    else:
        st.success("✅ Classifier loaded!")

        # PREPROCESS
        image = img.resize((224,224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = classifier.predict(image)[0]
        class_id = np.argmax(prediction)

        st.write(f"Hasil Prediksi: {class_id}")

# =============================================
# MODE YOLO DETEKSI
# =============================================

elif mode == "Deteksi (YOLO)":

    if not yolo_status:
        st.error("❌ YOLO tidak tersedia!")
    else:
        st.success("✅ YOLO loaded!")

        result = yolo_model.predict(np.array(img))
        annotated_frame = result[0].plot()

        st.image(annotated_frame, caption="Hasil Deteksi", width=500)
