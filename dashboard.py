import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tensorflow as tf

# ======================================================
# LOAD MODEL (cached biar tidak reload terus)
# ======================================================
@st.cache_resource
def initialize_models():
    detector = YOLO("Model/Elfii_Laporan 4.pt")
    classifier = tf.keras.models.load_model("Model/elfi_Laporan 2.h5")
    return detector, classifier

detector, classifier = initialize_models()

# ======================================================
# TITLE & SIDEBAR
# ======================================================
st.set_page_config(page_title="Dashboard Cerdas", layout="wide")
st.title("🧠 Sistem Deteksi & Klasifikasi Gambar")

mode = st.sidebar.radio(
    "Pilih Mode Analisa",
    ["🔍 Deteksi Objek (YOLO)", "🧾 Klasifikasi Gambar"]
)

uploaded = st.file_uploader("Upload file gambar:", type=["jpg", "jpeg", "png"])

# ======================================================
# JIKA ADA GAMBAR
# ======================================================
if uploaded:
    gambar = Image.open(uploaded)
    st.image(gambar, caption="Preview Gambar", use_column_width=True)

    # ------------------ MODE DETEKSI ------------------
    if mode == "🔍 Deteksi Objek (YOLO)":
        hasil = detector(gambar)
        annotated = hasil[0].plot()
        st.subheader("📌 Hasil Deteksi")
        st.image(annotated, use_column_width=True)

    # ------------------ MODE KLASIFIKASI --------------
    else:
        resize = gambar.resize((224, 224))
        arr = np.array(resize) / 255.0
        arr = np.expand_dims(arr, axis=0)

        score = classifier.predict(arr)
        pred_class = np.argmax(score)
        prob = np.max(score)

        st.subheader("📌 Hasil Klasifikasi")
        st.success(f"Prediksi kelas: **{pred_class}**")
        st.write(f"Akurasi probabilitas: **{prob:.4f}**")

else:
    st.info("Silakan upload gambar terlebih dahulu 👆")
