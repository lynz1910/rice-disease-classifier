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
# CLASS NAMES & INFO
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
    "Bacterial Leaf Blight": {
        "emoji": "🦠",
        "severity": "Tinggi",
        "color": "#ef4444",
        "description": "Penyakit bakteri yang menyebabkan layu dan kematian daun.",
        "treatment": "Gunakan varietas tahan, kelola air dengan baik, dan aplikasikan bakterisida jika diperlukan."
    },
    "Brown Spot": {
        "emoji": "🟤",
        "severity": "Sedang",
        "color": "#f97316",
        "description": "Penyakit jamur yang menyebabkan bercak coklat pada daun.",
        "treatment": "Perbaiki drainase, gunakan pupuk berimbang, dan aplikasikan fungisida."
    },
    "Healthy Rice Leaf": {
        "emoji": "✅",
        "severity": "Sehat",
        "color": "#22c55e",
        "description": "Daun padi dalam kondisi sehat tanpa tanda penyakit.",
        "treatment": "Pertahankan praktik budidaya yang baik untuk menjaga kesehatan tanaman."
    },
    "Leaf Blast": {
        "emoji": "💥",
        "severity": "Tinggi",
        "color": "#dc2626",
        "description": "Penyakit jamur serius yang menyebabkan lesion berbentuk berlian pada daun.",
        "treatment": "Gunakan varietas tahan, kelola nitrogen, dan aplikasikan fungisida sistemik."
    },
    "Leaf Scald": {
        "emoji": "🔥",
        "severity": "Sedang",
        "color": "#f59e0b",
        "description": "Penyakit bakteri yang menyebabkan garis-garis putih keabu-abuan pada daun.",
        "treatment": "Gunakan benih sehat, rotasi tanaman, dan aplikasikan bakterisida."
    },
    "Sheath Blight": {
        "emoji": "🍂",
        "severity": "Tinggi",
        "color": "#ea580c",
        "description": "Penyakit jamur yang menyerang batang dan pelepah daun.",
        "treatment": "Kurangi kepadatan tanaman, kelola air, dan gunakan fungisida."
    }
}

# =======================
# PATH (SUPPORT src/)
# =======================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "model.pth")
BG_PATH = os.path.join(ROOT_DIR, "background_sawah.png")

# =======================
# BACKGROUND
# =======================
def load_bg_base64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_base64 = load_bg_base64(BG_PATH)

# =======================
# ENHANCED CSS
# =======================
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {{
    font-family: 'Inter', sans-serif;
}}

[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: fixed;
    inset: 0;
    background: linear-gradient(135deg, rgba(0,0,0,0.7) 0%, rgba(34,139,34,0.4) 100%);
    backdrop-filter: blur(2px);
    pointer-events: none;
}}

section.main > div {{
    background: rgba(255,255,255,0.98);
    border-radius: 24px;
    padding: 40px;
    max-width: 800px;
    margin: 40px auto;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    animation: fadeIn 0.6s ease-in;
    position: relative;
    z-index: 1;
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

h1 {{
    color: #1a5f1a;
    font-weight: 700;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    text-align: center;
}}

.subtitle {{
    text-align: center;
    color: white;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}}

[data-testid="stFileUploader"] {{
    background: #4CAF50 !important;
    border: 2px dashed #2e7d32 !important;
    border-radius: 16px !important;
    padding: 30px !important;
    transition: all 0.3s ease !important;
}}

[data-testid="stFileUploader"] label {{
    color: #ffffff !important;
    font-weight: 600 !important;
}}

[data-testid="stFileUploader"] small {{
    color: #e8f5e9 !important;
}}

[data-testid="stFileUploader"] span {{
    color: #000000 !important;
}}

[data-testid="stFileUploader"] section {{
    background-color: #2e7d32 !important;
    border: 2px dashed #81c784 !important;
    border-radius: 8px !important;
}}

[data-testid="stFileUploader"] section div {{
    color: #ffffff !important;
}}

[data-testid="stFileUploader"] section button {{
    background-color: #1b5e20 !important;
    color: #ffffff !important;
    border: none !important;
}}

[data-testid="stFileUploader"]:hover {{
    border-color: #1b5e20 !important;
    background: #66bb6a !important;
    transform: translateY(-2px) !important;
}}

button[kind="primary"] {{
    background: linear-gradient(135deg, #4CAF50 0%, #2e7d32 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    height: 54px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    color: white !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(76,175,80,0.3) !important;
}}

button[kind="primary"]:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(76,175,80,0.4) !important;
}}

.result-card {{
    background: #ffffff;
    border-radius: 16px;
    padding: 28px;
    margin: 20px 0;
    border-left: 5px solid;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    animation: slideIn 0.5s ease-out;
}}

@keyframes slideIn {{
    from {{ opacity: 0; transform: translateX(-20px); }}
    to {{ opacity: 1; transform: translateX(0); }}
}}

.disease-title {{
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
}}

.severity-badge {{
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
    color: white;
}}

.info-section {{
    background: #f1f5f9;
    border-radius: 12px;
    padding: 18px;
    margin-top: 16px;
    border: 1px solid #e2e8f0;
}}

.info-label {{
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 10px;
    font-size: 1.05rem;
}}

.info-text {{
    color: #1e293b;
    line-height: 1.7;
    font-size: 0.95rem;
}}

.prob-item {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    transition: all 0.2s ease;
}}

.prob-item:hover {{
    background: #f8fafc;
    border-color: #cbd5e1;
    transform: translateX(4px);
}}

.prob-name {{
    font-weight: 600;
    color: #1e293b;
    font-size: 0.95rem;
}}

.prob-value {{
    font-weight: 700;
    font-size: 1.05rem;
    padding: 4px 12px;
    border-radius: 6px;
    background: #f1f5f9;
}}

.confidence-high {{ color: #16a34a; }}
.confidence-med {{ color: #ea580c; }}
.confidence-low {{ color: #64748b; }}

[data-testid="stImage"] {{
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin: 20px 0;
}}

.instructions {{
    background: rgba(59, 130, 246, 0.15);
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 16px;
    margin: 20px 0;
    backdrop-filter: blur(10px);
}}

.instructions-title {{
    font-weight: 600;
    color: white;
    margin-bottom: 8px;
    font-size: 1.1rem;
}}

.instructions-list {{
    color: white;
    margin-left: 20px;
    line-height: 1.8;
}}

[data-testid="stExpander"] {{
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
}}

[data-testid="stExpander"] summary {{
    color: white !important;
}}
</style>
""",
    unsafe_allow_html=True
)

# =======================
# LOAD MODEL
# =======================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ model.pth tidak ditemukan di {MODEL_PATH}")
        st.stop()
    model = get_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# =======================
# STREAMLIT UI (ENHANCED)
# =======================
st.markdown("<h1>🌾 Klasifikasi Penyakit Padi</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle' style='color: white;'>Sistem deteksi cerdas untuk mengidentifikasi penyakit pada daun padi</p>", unsafe_allow_html=True)

# Instructions
with st.expander("📖 Cara Menggunakan Aplikasi", expanded=False):
    st.markdown("""
    <div class='instructions-title' style='color: white;'>Langkah-langkah:</div>
    <ol class='instructions-list' style='color: white;'>
        <li>Ambil foto daun padi dengan pencahayaan yang baik</li>
        <li>Pastikan daun terlihat jelas tanpa blur</li>
        <li>Unggah gambar melalui tombol di bawah</li>
        <li>Klik tombol 'Analisis Gambar' untuk memulai deteksi</li>
        <li>Lihat hasil prediksi dan rekomendasi perawatan</li>
    </ol>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Unggah gambar daun padi (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Pilih gambar daun padi yang ingin dianalisis"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Gambar yang Diunggah", use_container_width=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🔍 Analisis Gambar", use_container_width=True)
    
    if predict_btn:
        with st.spinner("⏳ Menganalisis gambar..."):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            img = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                output = model(img)
                probs = torch.softmax(output, dim=1)[0]
            
            idx = probs.argmax().item()
            predicted_class = CLASS_NAMES[idx]
            confidence = probs[idx].item() * 100
            info = DISEASE_INFO[predicted_class]
            
            # Result Card - Using components instead of raw HTML
            st.markdown(f"""
<div class='result-card' style='border-left-color: {info['color']};'>
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 12px;'>
        <span style='font-size: 2.5rem;'>{info['emoji']}</span>
        <h2 style='color: {info['color']}; margin: 0; font-size: 1.8rem; font-weight: 700;'>{predicted_class}</h2>
    </div>
    <span class='severity-badge' style='background-color: {info['color']};'>Tingkat: {info['severity']}</span>
    <div style='margin-top: 16px; font-size: 1.3rem; font-weight: 600; color: #1e293b;'>
        Tingkat Kepercayaan: <span style='color: {info['color']};'>{confidence:.1f}%</span>
    </div>
    <div class='info-section'>
        <div class='info-label'>📋 Deskripsi:</div>
        <div class='info-text'>{info['description']}</div>
    </div>
    <div class='info-section'>
        <div class='info-label'>💊 Rekomendasi Perawatan:</div>
        <div class='info-text'>{info['treatment']}</div>
    </div>
</div>
""", unsafe_allow_html=True)
            
            # Detailed Probabilities
            st.markdown("### 📊 Analisis Detail Probabilitas")
            
            for cls, prob in zip(CLASS_NAMES, probs):
                prob_val = prob.item() * 100
                cls_info = DISEASE_INFO[cls]
                
                if prob_val >= 50:
                    conf_class = "confidence-high"
                elif prob_val >= 20:
                    conf_class = "confidence-med"
                else:
                    conf_class = "confidence-low"
                
                st.markdown(
                    f"""<div class='prob-item'>
                        <span class='prob-name'>{cls_info['emoji']} {cls}</span>
                        <span class='prob-value {conf_class}'>{prob_val:.2f}%</span>
                    </div>""",
                    unsafe_allow_html=True
                )
            
            # Disclaimer
            st.info("ℹ️ **Catatan:** Hasil ini adalah prediksi AI dan sebaiknya dikonfirmasi oleh ahli agronomi untuk diagnosis yang lebih akurat.")

else:
    st.markdown("""
    <div class='instructions'>
        <div class='instructions-title' style='color: white;'>👋 Selamat Datang!</div>
        <div class='info-text' style='color: white;'>
            Aplikasi ini menggunakan teknologi Deep Learning untuk membantu petani 
            mengidentifikasi penyakit pada tanaman padi secara cepat dan akurat. 
            Mulai dengan mengunggah gambar daun padi di atas.
        </div>
    </div>
    """, unsafe_allow_html=True)