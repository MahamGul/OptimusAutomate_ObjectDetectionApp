import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEURAL SCAN WEBAPP",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

  /* ── root reset ── */
  html, body, [data-testid="stAppViewContainer"] {
    background: #070b0f !important;
    color: #c8e6f5 !important;
  }
  [data-testid="stHeader"] { background: transparent !important; }
  [data-testid="stSidebar"] { background: #0d1117 !important; }

  /* ── typography defaults ── */
  *, p, li, div { font-family: 'Rajdhani', sans-serif; }
  code, pre, .mono { font-family: 'Share Tech Mono', monospace !important; }

  /* ── hero title ── */
  .hero-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: clamp(2rem, 6vw, 4.5rem);
    font-weight: 400;
    letter-spacing: 0.18em;
    color: #00e5ff;
    text-transform: uppercase;
    line-height: 1;
    margin: 0;
  }
  .hero-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.35em;
    color: #3a6980;
    text-transform: uppercase;
    margin-top: 0.4rem;
  }
  .hero-rule {
    border: none;
    border-top: 1px solid #00e5ff22;
    margin: 1.5rem 0;
  }

  /* ── panel ── */
  .panel {
    background: #0d1520;
    border: 1px solid #1a3545;
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
    position: relative;
  }
  .panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 40px; height: 2px;
    background: #00e5ff;
  }
  .panel-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.3em;
    color: #00e5ff;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
  }

  /* ── detection card ── */
  .det-card {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.55rem 0.75rem;
    margin-bottom: 0.5rem;
    background: #111d28;
    border-left: 2px solid #00e5ff;
    border-radius: 2px;
  }
  .det-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: #e0f4ff;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    flex: 1;
  }
  .det-conf {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: #00e5ff;
  }
  .det-bar-wrap {
    width: 60px;
    height: 4px;
    background: #1a2d3d;
    border-radius: 2px;
    overflow: hidden;
  }
  .det-bar {
    height: 4px;
    background: linear-gradient(90deg, #00e5ff, #0077ff);
    border-radius: 2px;
  }

  /* ── stat block ── */
  .stat-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
  }
  .stat-box {
    flex: 1;
    background: #111d28;
    border: 1px solid #1a3545;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    text-align: center;
  }
  .stat-num {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    color: #00e5ff;
    line-height: 1;
  }
  .stat-lbl {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: #3a6980;
    text-transform: uppercase;
    margin-top: 0.25rem;
  }

  /* ── Streamlit overrides ── */
  [data-testid="stFileUploader"] {
    border: 1px dashed #1a3545 !important;
    background: #0d1520 !important;
    border-radius: 4px !important;
  }
  .stButton > button {
    background: transparent !important;
    border: 1px solid #00e5ff !important;
    color: #00e5ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 0.15em !important;
    font-size: 0.8rem !important;
    border-radius: 2px !important;
    padding: 0.5rem 1.5rem !important;
    transition: background 0.2s !important;
  }
  .stButton > button:hover {
    background: #00e5ff18 !important;
  }
  [data-testid="stCameraInput"] {
    border: 1px dashed #1a3545 !important;
    background: #0d1520 !important;
    border-radius: 4px !important;
  }
  [data-testid="stImage"] img {
    border: 1px solid #1a2d3d;
    border-radius: 2px;
  }
  .stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0.25rem;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.2em !important;
    color: #3a6980 !important;
    background: #0d1520 !important;
    border: 1px solid #1a3545 !important;
    border-radius: 2px !important;
    padding: 0.4rem 1rem !important;
    text-transform: uppercase !important;
  }
  .stTabs [aria-selected="true"] {
    color: #00e5ff !important;
    border-color: #00e5ff !important;
    background: #0d1520 !important;
  }
  .stTabs [data-baseweb="tab-border"] { display: none !important; }

  /* ── scanline overlay (decorative) ── */
  .scanline-banner {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    color: #1a3545;
    text-transform: uppercase;
    border-top: 1px solid #1a3545;
    border-bottom: 1px solid #1a3545;
    padding: 0.3rem 0;
    margin-bottom: 1.5rem;
    overflow: hidden;
    white-space: nowrap;
  }

  /* ── success / empty states ── */
  .empty-hint {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #1a3545;
    letter-spacing: 0.15em;
    text-align: center;
    padding: 2rem 0;
    text-transform: uppercase;
  }
  .no-detections {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: #e5a000;
    letter-spacing: 0.1em;
    padding: 0.5rem;
    border-left: 2px solid #e5a000;
    background: #1a1400;
  }
</style>
""", unsafe_allow_html=True)


# ── Model load ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()


# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 0.5rem;">
  <p class="hero-title">NEURAL SCAN</p>
  <p class="hero-sub">YOLOv8 · Real-time Object Detection System · v8n</p>
</div>
<hr class="hero-rule"/>
<div class="scanline-banner">
  ▸ SYSTEM ONLINE &nbsp;·&nbsp; MODEL LOADED: yolov8n.pt &nbsp;·&nbsp; CLASSES: 80 &nbsp;·&nbsp; STATUS: READY FOR INPUT &nbsp;·&nbsp; INFERENCE ENGINE: ULTRALYTICS &nbsp;·&nbsp;
</div>
""", unsafe_allow_html=True)


# ── Helper: run detection ───────────────────────────────────────────────────────
def run_detection(image_np):
    t0 = time.time()
    results = model(image_np)
    elapsed = (time.time() - t0) * 1000

    annotated = results[0].plot(
        line_width=2,
        font_size=0.5,
        pil=False,
    )

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        detections.append({"label": label, "conf": conf})

    detections.sort(key=lambda x: x["conf"], reverse=True)
    return annotated, detections, elapsed


def render_results(annotated, detections, elapsed):
    """Render annotated image + detection cards side-by-side."""
    left, right = st.columns([3, 2], gap="medium")

    with left:
        st.markdown('<div class="panel-label">▸ Annotated output</div>', unsafe_allow_html=True)
        st.image(annotated, use_container_width=True)

    with right:
        # Stats row
        unique = len(set(d["label"] for d in detections))
        avg_conf = (sum(d["conf"] for d in detections) / len(detections) * 100) if detections else 0
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-box">
            <div class="stat-num">{len(detections)}</div>
            <div class="stat-lbl">objects</div>
          </div>
          <div class="stat-box">
            <div class="stat-num">{unique}</div>
            <div class="stat-lbl">classes</div>
          </div>
          <div class="stat-box">
            <div class="stat-num">{elapsed:.0f}</div>
            <div class="stat-lbl">ms</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="panel-label">▸ Detections</div>', unsafe_allow_html=True)

        if not detections:
            st.markdown('<div class="no-detections">⚠ NO OBJECTS DETECTED</div>', unsafe_allow_html=True)
        else:
            for d in detections:
                bar_pct = int(d["conf"] * 100)
                conf_str = f"{d['conf']:.2f}"
                st.markdown(f"""
                <div class="det-card">
                  <span class="det-label">{d['label'].upper()}</span>
                  <div class="det-bar-wrap"><div class="det-bar" style="width:{bar_pct}%"></div></div>
                  <span class="det-conf">{conf_str}</span>
                </div>
                """, unsafe_allow_html=True)


# ── Input tabs ─────────────────────────────────────────────────────────────────
tab_upload, tab_cam = st.tabs(["📂  UPLOAD IMAGE", "📷  WEBCAM CAPTURE"])


# ── Tab 1: Upload ──────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop an image file",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)

        with st.spinner(""):
            st.markdown('<p style="font-family:\'Share Tech Mono\',monospace;font-size:0.75rem;color:#00e5ff;letter-spacing:0.2em;">▸ RUNNING INFERENCE...</p>', unsafe_allow_html=True)
            annotated, detections, elapsed = run_detection(image_np)

        st.markdown("<hr style='border-color:#1a3545;margin:1.25rem 0'>", unsafe_allow_html=True)
        render_results(annotated, detections, elapsed)
    else:
        st.markdown('<div class="empty-hint">── awaiting image input ──</div>', unsafe_allow_html=True)


# ── Tab 2: Webcam ──────────────────────────────────────────────────────────────
with tab_cam:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="panel" style="margin-bottom:1rem;">
      <div class="panel-label">▸ Webcam capture</div>
      <p style="font-size:0.9rem;color:#3a6980;margin:0;">
        Point your camera at any scene and hit the shutter button.
        Detection runs the moment the photo is taken.
      </p>
    </div>
    """, unsafe_allow_html=True)

    cam_photo = st.camera_input("Take a photo", label_visibility="collapsed")
    if cam_photo:
        if st.button("🔄  RETAKE"):
            st.rerun()
    if cam_photo:
        image = Image.open(cam_photo).convert("RGB")
        image_np = np.array(image)

        with st.spinner(""):
            st.markdown('<p style="font-family:\'Share Tech Mono\',monospace;font-size:0.75rem;color:#00e5ff;letter-spacing:0.2em;">▸ SCANNING FRAME...</p>', unsafe_allow_html=True)
            annotated, detections, elapsed = run_detection(image_np)

        st.markdown("<hr style='border-color:#1a3545;margin:1.25rem 0'>", unsafe_allow_html=True)
        render_results(annotated, detections, elapsed)
    else:
        st.markdown('<div class="empty-hint">── awaiting camera input ──</div>', unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#1a3545;margin:3rem 0 1rem;">
<p style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;letter-spacing:0.25em;color:#1a3545;text-align:center;text-transform:uppercase;">
  Neural Scan · YOLOv8n · Ultralytics · Built with Streamlit
</p>
""", unsafe_allow_html=True)