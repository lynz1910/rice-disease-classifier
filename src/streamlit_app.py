import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import base64
from model import get_model

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="Klasifikasi Penyakit Padi",
    page_icon="🌾",
    layout="centered"
)

# =======================
# CLASS & INFO
# =======================
CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight"
]

DISEASE_INFO = {
    "Bacterial Leaf Blight": ("🦠", "#dc2626",
        "Penyakit bakteri yang menyebabkan layu daun.",
        "Gunakan varietas tahan dan bakterisida."),
    "Brown Spot": ("🟤", "#f97316",
        "Bercak coklat akibat jamur.",
        "Perbaiki drainase dan fungisida."),
    "Healthy Rice Leaf": ("✅", "#16a34a",
        "Daun sehat tanpa penyakit.",
        "Pertahankan perawatan."),
    "Leaf Blast": ("💥", "#b91c1c",
        "Jamur serius dengan lesi berlian.",
        "Kelola nitrogen dan fungisida."),
    "Leaf Scald": ("🔥", "#f59e0b",
        "Garis putih keabu-abuan.",
        "Benih sehat dan bakterisida."),
    "Sheath Blight": ("🍂", "#ea580c",
        "Jamur pelepah daun.",
        "Kurangi kepadatan tanaman.")
}

# =======================
# PATH
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "model.pth")
BG_PATH = os.path.join(ROOT_DIR, "background_sawah.png")

# =======================
# BACKGROUND BASE64
# =======================
def load_bg(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = load_bg(BG_PATH)

# =======================
# CSS (PALING AMAN)
# =======================
if bg:
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
.app-box {
    background: rgba(255,255,255,0.95);
    padding: 35px;
    border-radius: 20px;
    max-width: 850px;
    margin: auto;
    box-shadow: 0 15px 40px rgba(0,0,0,0.35);
}

h1 {
    text-align: center;
    color: #14532d;
}

.subtitle {
    text-align: center;
    color: #334155;
    margin-bottom: 25px;
}

.result {
    border-left: 6px solid;
    padding: 20px;
    border-radius: 14px;
    background: white;
    margin-top: 20px;
    color: black;
}

.prob {
    display: flex;
    justify-content: space-between;
    padding: 8px 14px;
    border-bottom: 1px solid #e5e7eb;
    color: black;
}
</style>
""", unsafe_allow_html=True)

# =======================
# LOAD MODEL (AMAN)
# =======================
try:
    cache_resource = st.cache_resource
except AttributeError:
    cache_resource = st.cache

@cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ model.pth tidak ditemukan")
        st.stop()

    model = get_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# =======================
# UI
# =======================
st.markdown("<div class='app-box'>", unsafe_allow_html=True)

st.markdown("<h1>🌾 Klasifikasi Penyakit Padi</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload gambar daun padi untuk dianalisis</div>", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Unggah gambar daun padi",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Gambar diunggah")

    _, col, _ = st.columns([1,2,1])
    with col:
        run = st.button("🔍 Analisis Gambar")

    if run:
        with st.spinner("Menganalisis..."):
            tfm = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])
            img = tfm(image).unsqueeze(0)

            with torch.no_grad():
                out = model(img)
                probs = torch.softmax(out, dim=1)[0]

            idx = probs.argmax().item()
            cls = CLASS_NAMES[idx]
            emoji, color, desc, treat = DISEASE_INFO[cls]
            conf = probs[idx].item() * 100

            st.markdown(f"""
            <div class="result" style="border-color:{color}">
                <h2>{emoji} {cls}</h2>
                <p><b>Kepercayaan:</b> {conf:.2f}%</p>
                <p><b>Deskripsi:</b> {desc}</p>
                <p><b>Perawatan:</b> {treat}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 Probabilitas")
            for c, p in zip(CLASS_NAMES, probs):
                st.markdown(
                    f"<div class='prob'><span>{c}</span><span>{p.item()*100:.2f}%</span></div>",
                    unsafe_allow_html=True
                )
else:
    st.info("👋 Silakan unggah gambar daun padi untuk memulai analisis.")

st.markdown("</div>", unsafe_allow_html=True)
