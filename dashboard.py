import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf

# ==============================================
# PATH MODEL
# ==============================================
clf_path = "model/elfi_Laporan 2.h5"         # model klasifikasi
yolo_path = "model/Elfii_Laporan 4.pt"       # model YOLO

# ==============================================
# LOAD MODEL KLASIFIKASI
# ==============================================
try:
    classifier = tf.keras.models.load_model(clf_path)
    clf_loaded = True
except Exception as e:
    classifier = None
    clf_loaded = False
    st.write(e)

# ==============================================
# LOAD MODEL YOLO
# ==============================================
try:
    yolo_model = YOLO(yolo_path)
    yolo_loaded = True
except Exception as e:
    yolo_model = None
    yolo_loaded = False
    st.write(e)

# ==============================================
# STREAMLIT UI
# ==============================================
st.sidebar.title("⚙ Panel Kontrol")
mode = st.sidebar.selectbox("Mode Analisis:", ["Klasifikasi", "Deteksi YOLO"])

uploaded_image = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption="Gambar Input", width=500)

# ==============================================
# MODE KLASIFIKASI
# ==============================================
if mode == "Klasifikasi":

    if not clf_loaded:
        st.error("❌ Model klasifikasi TIDAK ditemukan!")
    else:
        st.success("✅ Model klasifikasi berhasil diload!")

        # Pre-process gambar
        image = img.resize((224,224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        pred = classifier.predict(image)[0]
        kelas = np.argmax(pred)

        st.info(f"Hasil Prediksi Kelas: {kelas}")

# ==============================================
# MODE YOLO DETEKSI
# ==============================================
elif mode == "Deteksi YOLO":

    if not yolo_loaded:
        st.error("❌ Model YOLO TIDAK ditemukan!")
    else:
        st.success("✅ Model YOLO berhasil diload!")

        result = yolo_model.predict(np.array(img))
        annotated = result[0].plot()

        st.image(annotated, caption="Hasil Deteksi YOLO", width=500)
