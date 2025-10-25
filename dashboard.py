import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================
# Load Models
# ==========================
@st.cache_resource()
def load_models():
    yolo_model = YOLO("Model/Elfii_Laporan 4.pt")  # YOLO deteksi objek
    classifier = tf.keras.models.load_model("Model/elfi_Laporan 2.h5")  # klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        results = yolo_model(img)
        result_img = results[0].plot()  # annotate bounding boxes
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        img_resized = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)

        st.success(f"✅ **Prediksi kelas:** {class_index}")
        st.info(f"📌 Probabilitas: {np.max(prediction):.4f}")
