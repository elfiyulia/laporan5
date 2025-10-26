import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import os, glob

# Try optional YOLO
try:
    from ultralytics import YOLO
    YOLO_READY = True
except:
    YOLO_READY = False

# =====================
# PAGE SETTINGS
# =====================
st.set_page_config(
    page_title="🔍 Vision Inspector",
    page_icon="🚗",
    layout="wide"
)

# =====================
# CUSTOM CSS STYLE
# =====================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(140deg, #ffc2d1 0%, #ff8fab 50%, #ff5d8f 100%);
    color: white;
}
.header-box {
    background: rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 18px;
    text-align:center;
    margin-bottom:20px;
}
.card-output {
    background: rgba(255,255,255,0.18);
    padding: 15px;
    border-radius:16px;
    margin-top:10px;
}
footer {
    text-align:center;
    margin-top:40px;
    opacity:0.9;
}
</style>
""", unsafe_allow_html=True)

# =====================
# MODEL FOLDER
# =====================
FOLDER = "model"

def search_model(extension):
    files = glob.glob(os.path.join(FOLDER, extension))
    return files[0] if files else None

# =====================
# LOAD CLASSIFIER
# =====================
@st.cache_resource
def get_classifier():
    path = search_model("*.h5")
    if not path: return None, "Classifier tidak ditemukan!"
    try:
        mdl = tf.keras.models.load_model(path)
        return mdl, f"Loaded: {os.path.basename(path)}"
    except Exception as e:
        return None, str(e)

# =====================
# LOAD YOLO
# =====================
@st.cache_resource
def get_detector():
    if not YOLO_READY: return None, "YOLO tidak tersedia!"
    path = search_model("*.pt")
    if not path: return None, "Model YOLO (.pt) kosong!"
    try:
        mdl = YOLO(path)
        return mdl, f"Loaded: {os.path.basename(path)}"
    except Exception as e:
        return None, str(e)

# Initialize
clf, clf_info = get_classifier()
det, det_info = get_detector()

# =====================
# LABEL DICTIONARY
# =====================
kategori = ["Bike","Car","Motorcycle","Plane","Ship",
            "grontol","lanting","lumpia","putu_ayu","wajik"]

detail = {
    "Bike":"Sepeda roda dua tanpa mesin",
    "Car":"Mobil roda empat untuk transportasi darat",
    "Motorcycle":"Sepeda motor bermesin",
    "Plane":"Pesawat transportasi udara",
    "Ship":"Kapal laut",
    "grontol":"Kudapan jagung tradisional Jawa",
    "lanting":"Camilan gorengan berbentuk cincin",
    "lumpia":"Makanan berisi sayur/daging dengan kulit tipis",
    "putu_ayu":"Kue kukus hijau lembut",
    "wajik":"Kue ketan legit manis"
}

# =====================
# IMAGE PREPROCESS
# =====================
def prepare(img, model):
    target = model.input_shape[1:3]
    img = img.resize(target)
    arr = img_to_array(img)
    arr = np.expand_dims(arr,0)/255.0
    return arr

def classify(model, image):
    arr = prepare(image, model)
    pred = model.predict(arr)
    idx = np.argmax(pred)
    return kategori[idx], np.max(pred)

# =====================
# HEADER UI
# =====================
st.markdown("<div class='header-box'><h1>🚗 Vision Inspector Dashboard</h1><h5>Analisis Kendaraan & Jajanan Tradisional</h5></div>", unsafe_allow_html=True)

# =====================
# SIDEBAR
# =====================
st.sidebar.subheader("🛠 Panel Kontrol")

mode = st.sidebar.selectbox("Mode Analisis:", 
                            ["Klasifikasi", "Deteksi Objek", "Filter Visual", "Dominasi Warna"])

st.sidebar.info(clf_info)
st.sidebar.info(det_info)

# =====================
# UPLOAD AREA
# =====================
upload = st.file_uploader("Unggah gambar:", type=["jpg","jpeg","png"])

if upload:
    img = Image.open(upload).convert("RGB")
    st.image(img, caption="Gambar Input", use_container_width=True)

    st.markdown("---")

    # ====== KLASIFIKASI ======
    if mode == "Klasifikasi":
        if clf is None:
            st.error("Model tidak tersedia!")
        else:
            lbl, prob = classify(clf, img)
            info = detail.get(lbl, "Tidak ditemukan deskripsi.")

            st.markdown(f"""
            <div class='card-output'>
            <h2>{lbl}</h2>
            <b>Confidence:</b> {prob*100:.2f}%<br>
            <b>Info:</b> {info}
            </div>
            """, unsafe_allow_html=True)

    # ====== DETEKSI ======
    elif mode == "Deteksi Objek":
        if det is None:
            st.error("YOLO tidak dapat dijalankan!")
        else:
            res = det(img)
            st.image(res[0].plot(), caption="Output YOLO", use_container_width=True)

    # ====== FILTER ======
    elif mode == "Filter Visual":
        pilih = st.selectbox("Pilihan filter:", ["Asli","Hitam Putih","Keburaman","Garis Tepi"])
        if pilih == "Hitam Putih": out = ImageOps.grayscale(img)
        elif pilih == "Keburaman": out = img.filter(ImageFilter.GaussianBlur(4))
        elif pilih == "Garis Tepi": out = img.filter(ImageFilter.FIND_EDGES)
        else: out = img
        st.image(out, caption=f"Mode: {pilih}", use_container_width=True)

    # ====== WARNA ======
    elif mode == "Dominasi Warna":
        shrink = img.resize((60,60))
        arr = np.array(shrink).reshape(-1,3)
        unique, count = np.unique(arr, axis=0, return_counts=True)
        idx = np.argsort(-count)[:4]
        dom = unique[idx]

        st.write("Warna paling dominan:")
        cols = st.columns(4)
        for i, c in enumerate(dom):
            hx = '#%02x%02x%02x'%tuple(c)
            cols[i].markdown(f"<div style='background:{hx};height:65px;border-radius:10px;'></div>", unsafe_allow_html=True)
            cols[i].write(hx)

else:
    st.warning("Silahkan upload gambar terlebih dahulu.")

# =====================
# FOOTER
# =====================
st.markdown("<footer>📌 Vision Inspector — dikembangkan oleh Elfi</footer>", unsafe_allow_html=True)
