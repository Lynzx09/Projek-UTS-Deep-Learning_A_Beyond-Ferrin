import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os
import io
import time

# ── Konfigurasi ───────────────────────────────────────────────────
IMG_SIZE    = 128
MODEL_PATH  = 'BestModel_VGG-16_Beyond Ferrin.h5' 
CLASS_NAMES = [
    'GIGI BERKARANG',
    'GIGI BERLUBANG',
    'GIGI HYPODONTIA',
    'MOUTH ULCER',
]
CLASS_INFO = {
    'GIGI BERKARANG': {
        'icon' : '🦷',
        'color': '#E8A838',
        'desc' : 'Karang gigi (kalkulus) adalah endapan mineral keras yang '
                 'terbentuk dari plak yang mengeras. Berwarna kuning hingga '
                 'coklat, biasanya muncul di pangkal gigi dekat gusi.',
        'saran': 'Segera lakukan scaling ke dokter gigi. Sikat gigi 2× sehari '
                 'dan gunakan benang gigi secara rutin.',
    },
    'GIGI BERLUBANG': {
        'icon' : '🕳️',
        'color': '#E05252',
        'desc' : 'Karies gigi adalah kerusakan struktur gigi akibat aktivitas '
                 'bakteri yang mengurai gula menjadi asam, merusak email gigi '
                 'hingga membentuk lubang.',
        'saran': 'Konsultasikan ke dokter gigi untuk penambalan. Kurangi '
                 'konsumsi makanan manis dan asam.',
    },
    'GIGI HYPODONTIA': {
        'icon' : '🔍',
        'color': '#5B8CDB',
        'desc' : 'Hypodontia adalah kondisi di mana seseorang memiliki jumlah '
                 'gigi yang kurang dari normal (kehilangan 1–5 gigi permanen) '
                 'akibat faktor genetik atau perkembangan.',
        'saran': 'Konsultasikan dengan dokter gigi atau ortodontis untuk '
                 'evaluasi dan rencana perawatan (implan, jembatan, atau kawat).',
    },
    'MOUTH ULCER': {
        'icon' : '💊',
        'color': '#7B5EA7',
        'desc' : 'Sariawan (mouth ulcer / aphthous ulcer) adalah luka kecil '
                 'berbentuk bulat/oval di jaringan lunak mulut. Umumnya '
                 'disebabkan stres, defisiensi nutrisi, atau cedera ringan.',
        'saran': 'Hindari makanan pedas dan asam. Gunakan obat kumur antiseptik. '
                 'Konsultasikan ke dokter jika tidak sembuh dalam 2 minggu.',
    },
}

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title='DentalVision — Klasifikasi Kondisi Gigi',
    page_icon='🦷',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default streamlit header/footer */
#MainMenu, footer, header { visibility: hidden; }

/* App background */
.stApp {
    background: #F7F5F0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1A1A2E;
    border-right: 1px solid #2D2D4E;
}
[data-testid="stSidebar"] * {
    color: #E8E8F0 !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #C8B8E8 !important;
    font-family: 'DM Serif Display', serif !important;
}

/* Navbar / Header custom */
.app-header {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
    padding: 2rem 2.5rem 1.5rem;
    border-radius: 0 0 24px 24px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.15);
}
.app-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #FFFFFF;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-header p {
    color: #A8B8D8;
    margin: 0.3rem 0 0;
    font-size: 0.95rem;
    font-weight: 300;
}

/* Cards */
.card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #EAEAEA;
    margin-bottom: 1rem;
}
.result-card {
    border-radius: 16px;
    padding: 1.8rem;
    margin-top: 1rem;
    animation: slideUp 0.4s ease;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Confidence bar */
.conf-bar-wrap {
    background: #F0F0F0;
    border-radius: 99px;
    height: 10px;
    margin: 4px 0 2px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}

/* Label kelas */
.class-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Metric box */
.metric-box {
    background: #F7F5F0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    border: 1px solid #E8E4DC;
}
.metric-box .val {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'DM Serif Display', serif;
    line-height: 1;
}
.metric-box .lbl {
    font-size: 0.75rem;
    color: #888;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border-radius: 16px;
    border: 2px dashed #D0C8BC;
    padding: 1rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: #0F3460;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #0F3460, #1A1A2E);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.8rem;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.3px;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(15,52,96,0.35);
}

/* Divider */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #1A1A2E;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #E8E4DC;
}

/* Disclaimer */
.disclaimer {
    background: #FFF8E8;
    border-left: 4px solid #E8A838;
    border-radius: 0 12px 12px 0;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #7A6030;
    margin-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load Model ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        return None
    
    # Fix kompatibilitas Keras versi berbeda
    # Error: 'quantization_config' tidak dikenal di versi lokal
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Conv2D, DepthwiseConv2D

    class CompatDense(Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(*args, **kwargs)

    class CompatConv2D(Conv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(*args, **kwargs)

    class CompatDepthwiseConv2D(DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(*args, **kwargs)

    custom_objects = {
        'Dense'           : CompatDense,
        'Conv2D'          : CompatConv2D,
        'DepthwiseConv2D' : CompatDepthwiseConv2D,
    }

    return tf.keras.models.load_model(path, custom_objects=custom_objects)
# ── Helper Functions ──────────────────────────────────────────────
def preprocess_image(img: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    img = img.convert('RGB').resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(model, img: Image.Image):
    arr   = preprocess_image(img)
    probs = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]) * 100, probs

def conf_color(conf: float) -> str:
    if conf >= 80: return '#27AE60'
    if conf >= 55: return '#E8A838'
    return '#E05252'

def render_prob_bars(probs):
    for i, cls in enumerate(CLASS_NAMES):
        pct   = float(probs[i]) * 100
        color = CLASS_INFO[cls]['color']
        icon  = CLASS_INFO[cls]['icon']
        st.markdown(f"""
        <div style="margin-bottom:10px">
            <div style="display:flex;justify-content:space-between;
                        font-size:0.82rem;margin-bottom:3px">
                <span>{icon} {cls}</span>
                <span style="font-weight:600;color:{color}">{pct:.1f}%</span>
            </div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill"
                     style="width:{pct}%;background:{color}"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦷 DentalVision")
    st.markdown("---")
    st.markdown("### Tentang Aplikasi")
    st.markdown(
        "Aplikasi klasifikasi kondisi gigi dan mulut berbasis **Deep Learning** "
        "menggunakan arsitektur CNN.\n\n"
        "Model dilatih dari nol (*from scratch*) menggunakan dataset gambar "
        "klinis yang dikumpulkan secara mandiri."
    )
    st.markdown("---")
    st.markdown("### 4 Kelas Kondisi")
    for cls, info in CLASS_INFO.items():
        st.markdown(f"{info['icon']} **{cls}**")
    st.markdown("---")
    st.markdown("### Info Model")

    model = load_model(MODEL_PATH)
    if model is not None:
        total_params = model.count_params()
        st.markdown(f"""
        | Atribut | Nilai |
        |---|---|
        | Arsitektur | VGG-16 |
        | Input Size | {IMG_SIZE}×{IMG_SIZE} px |
        | Parameter | {total_params:,} |
        | Kelas | {len(CLASS_NAMES)} |
        | Weights | From Scratch |
        """)
        st.success("✅ Model berhasil dimuat")
    else:
        st.error(f"❌ Model tidak ditemukan:\n`{MODEL_PATH}`")
        st.info("Pastikan file `.h5` ada di direktori yang sama dengan `app.py`")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#888;text-align:center'>"
        "UTS Pembelajaran Mendalam<br>Universitas Atma Jaya Yogyakarta<br>2025/2026"
        "</div>",
        unsafe_allow_html=True
    )


# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🦷 DentalVision</h1>
    <p>Sistem Klasifikasi Kondisi Gigi & Mulut — Deep Learning CNN</p>
</div>
""", unsafe_allow_html=True)


# ── Main Content ──────────────────────────────────────────────────
if model is None:
    st.error(
        f"**Model tidak ditemukan.** Pastikan file `{MODEL_PATH}` "
        "berada di folder yang sama dengan `app.py`."
    )
    st.stop()

tab1, tab2 = st.tabs(["🔍 Prediksi Gambar", "📊 Informasi Kelas"])

# ════════════════════════════════
# TAB 1 — PREDIKSI
# ════════════════════════════════
with tab1:
    col_upload, col_result = st.columns([1, 1], gap='large')

    with col_upload:
        st.markdown('<div class="section-title">Upload Gambar</div>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Pilih gambar kondisi gigi (JPG / PNG / WEBP)",
            type=['jpg', 'jpeg', 'png', 'webp'],
            label_visibility='collapsed'
        )

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption='Gambar yang diupload',
                     use_column_width=True)

            w, h = img.size
            st.markdown(f"""
            <div class="card" style="margin-top:0.8rem">
                <div style="display:flex;gap:1rem;justify-content:space-around">
                    <div class="metric-box">
                        <div class="val">{w}</div>
                        <div class="lbl">Lebar (px)</div>
                    </div>
                    <div class="metric-box">
                        <div class="val">{h}</div>
                        <div class="lbl">Tinggi (px)</div>
                    </div>
                    <div class="metric-box">
                        <div class="val">{img.mode}</div>
                        <div class="lbl">Mode</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            predict_btn = st.button("🔍 Analisis Gambar", use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:2rem 1rem;color:#AAA">
                <div style="font-size:3rem">📷</div>
                <div style="margin-top:0.5rem;font-size:0.9rem">
                    Upload gambar kondisi gigi untuk memulai analisis
                </div>
            </div>
            """, unsafe_allow_html=True)
            predict_btn = False

    with col_result:
        st.markdown('<div class="section-title">Hasil Analisis</div>',
                    unsafe_allow_html=True)

        if uploaded and predict_btn:
            with st.spinner('Menganalisis gambar...'):
                time.sleep(0.4)
                pred_class, confidence, probs = predict(model, img)

            info  = CLASS_INFO[pred_class]
            color = info['color']
            icon  = info['icon']
            ccolor = conf_color(confidence)

            # Result card
            st.markdown(f"""
            <div class="result-card card"
                 style="border-left: 5px solid {color}">
                <div class="class-badge"
                     style="background:{color}22;color:{color}">
                    {icon} Hasil Deteksi
                </div>
                <div style="font-family:'DM Serif Display',serif;
                            font-size:1.6rem;color:#1A1A2E;
                            margin:0.3rem 0 0.1rem">
                    {pred_class}
                </div>
                <div style="font-size:0.88rem;color:#666;
                            margin-bottom:1rem">
                    {icon} {icon} {icon}
                </div>
                <div style="display:flex;align-items:center;gap:0.8rem">
                    <div style="font-size:2.2rem;font-weight:700;
                                font-family:'DM Serif Display',serif;
                                color:{ccolor}">
                        {confidence:.1f}%
                    </div>
                    <div style="font-size:0.82rem;color:#888">
                        confidence<br>level
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Deskripsi & saran
            st.markdown(f"""
            <div class="card" style="margin-top:0.5rem">
                <div style="font-weight:600;margin-bottom:0.4rem;
                            color:#1A1A2E">📋 Deskripsi</div>
                <div style="font-size:0.88rem;color:#555;
                            line-height:1.6">
                    {info['desc']}
                </div>
                <div style="font-weight:600;margin:0.8rem 0 0.4rem;
                            color:#1A1A2E">💡 Saran</div>
                <div style="font-size:0.88rem;color:#555;
                            line-height:1.6">
                    {info['saran']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown('<div class="section-title" style="margin-top:1rem">'
                        'Probabilitas Semua Kelas</div>',
                        unsafe_allow_html=True)
            render_prob_bars(probs)


        elif not uploaded:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#BBB">
                <div style="font-size:3rem">📊</div>
                <div style="margin-top:0.5rem;font-size:0.9rem">
                    Hasil analisis akan muncul di sini
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════
# TAB 2 — INFO KELAS
# ════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Informasi 4 Kelas Kondisi</div>',
                unsafe_allow_html=True)

    for i in range(0, len(CLASS_NAMES), 2):
        cols = st.columns(2, gap='medium')
        for j, col in enumerate(cols):
            if i + j >= len(CLASS_NAMES):
                break
            cls  = CLASS_NAMES[i + j]
            info = CLASS_INFO[cls]
            with col:
                st.markdown(f"""
                <div class="card"
                     style="border-top: 4px solid {info['color']}">
                    <div style="font-size:2rem">{info['icon']}</div>
                    <div style="font-family:'DM Serif Display',serif;
                                font-size:1.15rem;font-weight:600;
                                color:#1A1A2E;margin:0.4rem 0 0.6rem">
                        {cls}
                    </div>
                    <div style="font-size:0.85rem;color:#555;
                                line-height:1.65;margin-bottom:0.8rem">
                        {info['desc']}
                    </div>
                    <div style="font-size:0.82rem;background:{info['color']}15;
                                border-radius:8px;padding:0.6rem 0.8rem;
                                color:{info['color']};font-weight:500">
                        💡 {info['saran']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Model comparison table
    st.markdown('<div class="section-title">Arsitektur Model yang Dilatih</div>',
                unsafe_allow_html=True)
    df_model = {
        'Arsitektur'  : ['Custom CNN', 'VGG-16', 'MobileNet'],
        'Tipe Weights': ['From Scratch', 'From Scratch', 'From Scratch'],
        'Keterangan'  : [
            'CNN buatan sendiri, 4 blok konvolusi + GAP',
            'Arsitektur VGG-16 (Simonyan & Zisserman, 2014)',
            'Depthwise Separable Conv (Howard et al., 2017)',
        ],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(df_model), use_container_width=True,
                 hide_index=True)
