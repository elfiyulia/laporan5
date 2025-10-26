import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import glob, os

# Optional YOLO
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="🚗🏍️ Vehicle & Food Vision",
    layout="wide",
    page_icon="✨"
)

# ======================
# CSS CUSTOM
# ======================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #ffb3c6 0%, #ff7eb9 40%, #ff5196 100%);
    color: #fff;
    font-family: 'Poppins', sans-serif;
}
.title-box {
    background: rgba(0,0,0,0.5);
    padding: 20px;
    border-radius: 20px;
    text-align:center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}
.result-card {
    background: rgba(255,255,255,0.15);
    padding: 18px;
    border-radius: 16px;
    margin-bottom: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    transition: transform 0.3s;
}
.result-card:hover {
    transform: scale(1.03);
}
footer {
    position: fixed;
    left: 10px;
    bottom: 10px;
    color:#fff;
    font-size: 14px;
    opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)

# ======================
# HELPER FUNCTIONS
# ======================
MODEL_FOLDER = "model"

def find_first(pattern):
    files = glob.glob(os.path.join(MODEL_FOLDER, pattern))
    return files[0] if files else None

@st.cache_resource
def load_classifier():
    path = find_first("*.h5")
    if not path: 
        return None, "Model .h5 tidak ditemukan dalam folder model/"
    try:
        model = tf.keras.models.load_model(path)
        return model, f"Dimuat dari {path}"
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_yolo():
    if not ULTRALYTICS_AVAILABLE:
        return None, "Ultralytics tidak tersedia di server ini"
    path = find_first("*.pt")
    if not path:
        return None, "Model .pt tidak ditemukan dalam folder model/"
    try:
        model = YOLO(path)
        return model, f"Dimuat dari {path}"
    except Exception as e:
        return None, str(e)

classifier, cls_info = load_classifier()
yolo, yolo_info = load_yolo()

# ======================
# LABEL DATA
# ======================
class_names = ["Bike","Car","Motorcycle","Plane","Ship",
               "grontol","lanting","lumpia","putu_ayu","wajik"]

label_info = {
    "Bike":"Kendaraan roda dua dengan pedal",
    "Car":"Kendaraan roda empat untuk transportasi umum",
    "Motorcycle":"Kendaraan roda dua dengan mesin",
    "Plane":"Alat transportasi udara",
    "Ship":"Transportasi laut",
    "grontol":"Makanan tradisional dari jagung",
    "lanting":"Cemilan goreng berbentuk lingkaran",
    "lumpia":"Makanan dengan isian sayur/daging",
    "putu_ayu":"Kue lembut berwarna hijau",
    "wajik":"Kue manis berbahan ketan dan gula merah"
}

# ======================
# PREPROCESS
# ======================
def preprocess_image(img, model):
    shape = model.input_shape[1:3]
    img_resized = img.resize(shape)
    arr = image.img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0)/255.0
    return arr

def predict_image(model, pil_img):
    arr = preprocess_image(pil_img, model)
    preds = model.predict(arr)
    idx = np.argmax(preds)
    label = class_names[idx] if idx < len(class_names) else "unknown"
    conf = float(np.max(preds))
    return label, conf

# ======================
# UI HEADER
# ======================
st.markdown("<div class='title-box'><h1>Vision Dashboard</h1><h4>Analisis Kendaraan & Makanan Tradisional</h4></div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Pengaturan")
if classifier: 
    st.sidebar.success(f"✅ Classifier aktif: {cls_info}")
else: 
    st.sidebar.warning("⚠️ Classifier belum dimuat")

if yolo: 
    st.sidebar.success(f"✅ YOLO aktif: {yolo_info}")
else: 
    st.sidebar.info("YOLO opsional")

mode = st.sidebar.radio("Pilih Mode:", ["Klasifikasi", "Deteksi Objek", "Filter Gambar", "Analisis Warna"])

# ======================
# UPLOAD IMAGE
# ======================
uploaded = st.file_uploader("📤 Unggah gambar:", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Gambar diunggah", use_container_width=True)
    st.markdown("---")

    # ===== KLASIFIKASI =====
    if mode=="Klasifikasi":
        if classifier is None: 
            st.error("Model classifier belum dimuat.")
        else:
            label, conf = predict_image(classifier, img)
            desc = label_info.get(label,"—")
            st.markdown(f"""
            <div class='result-card'>
            <h3>{label} ({conf*100:.2f}%)</h3>
            <b>Deskripsi:</b> {desc}
            </div>
            """, unsafe_allow_html=True)

    # ===== DETEKSI OBJEK =====
    elif mode=="Deteksi Objek":
        if yolo is None: 
            st.error("Model YOLO tidak aktif.")
        else:
            result = yolo(img)
            st.image(result[0].plot(), caption="Hasil Deteksi", use_container_width=True)

    # ===== FILTER =====
    elif mode=="Filter Gambar":
        filter_opt = st.selectbox("Pilih filter:", ["Asli","Grayscale","Blur","Sharpen","Edge"])
        out = img

        if filter_opt=="Grayscale": out = ImageOps.grayscale(img)
        elif filter_opt=="Blur": out = img.filter(ImageFilter.GaussianBlur(radius=3))
        elif filter_opt=="Sharpen": out = img.filter(ImageFilter.UnsharpMask(radius=3))
        elif filter_opt=="Edge": out = img.filter(ImageFilter.FIND_EDGES)

        st.image(out, caption=f"Filter: {filter_opt}", use_container_width=True)

    # ===== WARNA =====
    elif mode=="Analisis Warna":
        small = img.resize((120,120))
        arr = np.array(small).reshape(-1,3)
        uniq, counts = np.unique((arr//32)*32, axis=0, return_counts=True)
        top = uniq[np.argsort(-counts)[:5]]
        st.write("Warna dominan:")
        cols = st.columns(5)
        for i,c in enumerate(top):
            hexc = '#%02x%02x%02x'%tuple(c)
            cols[i].markdown(f"<div style='background:{hexc};height:80px;border-radius:12px;'></div>", unsafe_allow_html=True)
            cols[i].write(hexc)

else:
    st.info("📁 Unggah gambar kendaraan atau makanan tradisional untuk mulai analisis.")

# ======================
# FOOTER
# ======================
st.markdown("<footer>📌 Dashboard — by Elfi Yulia</footer>", unsafe_allow_html=True)
