import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

st.set_page_config(page_title="Deteksi Kondisi Buah & Sayur", page_icon="🌿", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0f1117; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .main-title { font-family: Georgia, serif; font-size: 2.2rem; font-weight: 700; color: #e6edf3; margin-bottom: 0; }
    .sub-title { font-family: monospace; font-size: 0.8rem; color: #7d8590; letter-spacing: 2px; text-transform: uppercase; margin-top: 4px; }
    .stage-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; }
    .stage-label { font-family: monospace; font-size: 0.7rem; color: #7d8590; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 6px; }
    .result-main { font-size: 1.5rem; font-weight: 700; color: #e6edf3; margin: 4px 0; }
    .result-healthy { color: #3fb950; }
    .result-rotten { color: #f85149; }
    .confidence-bar-bg { background: #21262d; border-radius: 4px; height: 6px; margin-top: 8px; }
    .confidence-bar-fill { height: 6px; border-radius: 4px; }
    .engine-badge { display: inline-block; font-family: monospace; font-size: 0.68rem; background: #21262d; border: 1px solid #30363d; border-radius: 4px; padding: 2px 8px; color: #8b949e; margin-top: 8px; }
    .prob-row { display: flex; justify-content: space-between; align-items: center; margin: 4px 0; font-family: monospace; font-size: 0.8rem; }
    .select-hint { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 2rem; text-align: center; color: #7d8590; font-family: monospace; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌿 Deteksi Kondisi Buah & Sayur</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Pure ML · OpenCV · XGBoost / SVM · Pilih Buah Terlebih Dahulu</p>', unsafe_allow_html=True)
st.markdown("<hr style='border-color:#21262d; margin: 1rem 0 1.5rem'>", unsafe_allow_html=True)

# ==============================================================================
# LOAD MODEL
# ==============================================================================
@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load('model.pkl')
        for f in pipeline['disease_models']:
            if pipeline['disease_models'][f]['type'] == 'tree':
                if hasattr(pipeline['disease_models'][f]['model'], 'set_params'):
                    pipeline['disease_models'][f]['model'].set_params(device='cpu')
        return pipeline
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

ml_pipeline = load_pipeline()
if ml_pipeline is None:
    st.stop()

AVAILABLE_FRUITS = sorted(ml_pipeline['disease_models'].keys())

# ==============================================================================
# FEATURE EXTRACTION
# ==============================================================================
IMG_SIZE = 128
RADIUS = 3
N_POINTS = 8 * RADIUS

def preprocess_to_studio_style(img_bgr):
    h, w = img_bgr.shape[:2]
    mask_gc = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    margin_x, margin_y = w // 6, h // 6
    rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
    try:
        cv2.grabCut(img_bgr, mask_gc, rect, bgd, fgd, 10, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest)
            pad = 20
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(w, x+cw+pad), min(h, y+ch+pad)
            img_bgr = img_bgr[y1:y2, x1:x2]
    except:
        pass
    img_bgr = cv2.resize(img_bgr, (256, 256))
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    img_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    mask_gc2 = np.zeros((256, 256), np.uint8)
    bgd2 = np.zeros((1, 65), np.float64)
    fgd2 = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, mask_gc2, (16,16,224,224), bgd2, fgd2, 5, cv2.GC_INIT_WITH_RECT)
        fg2 = np.where((mask_gc2 == 2) | (mask_gc2 == 0), 0, 255).astype(np.uint8)
        fg2 = cv2.morphologyEx(fg2, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        white_bg = np.ones_like(img_bgr) * 255
        fg2_3ch = cv2.cvtColor(fg2, cv2.COLOR_GRAY2BGR) // 255
        img_bgr = (img_bgr * fg2_3ch + white_bg * (1 - fg2_3ch)).astype(np.uint8)
    except:
        pass
    return img_bgr

@st.cache_data
def extract_advanced_features_segmented(image_array):
    img_resized = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    h, s, v = cv2.split(hsv)
    mask_grabcut = np.zeros(img_resized.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    margin = IMG_SIZE // 8
    rect = (margin, margin, IMG_SIZE - 2*margin, IMG_SIZE - 2*margin)
    try:
        cv2.grabCut(img_resized, mask_grabcut, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask_grabcut == 2) | (mask_grabcut == 0), 0, 255).astype(np.uint8)
        if mask.sum() < (IMG_SIZE * IMG_SIZE * 0.05 * 255): raise ValueError()
    except:
        _, thresh_s = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh_gray = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_or(thresh_s, thresh_gray)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        mask.fill(255)
        final_contour = np.array([[[0,0]], [[0,IMG_SIZE]], [[IMG_SIZE,IMG_SIZE]], [[IMG_SIZE,0]]])
    else:
        largest_contour = max(contours, key=cv2.contourArea)
        final_contour = cv2.convexHull(largest_contour)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [final_contour], -1, 255, thickness=cv2.FILLED)
    hist_h = cv2.calcHist([hsv], [0], mask, [32], [0, 256]).flatten()
    hist_s = cv2.calcHist([hsv], [1], mask, [32], [0, 256]).flatten()
    hist_a = cv2.calcHist([lab], [1], mask, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([lab], [2], mask, [32], [0, 256]).flatten()
    color_hist = np.concatenate([hist_h, hist_s, hist_a, hist_b])
    color_hist /= (color_hist.sum() + 1e-7)
    channels = [h, s, v, lab[:,:,0], lab[:,:,1], lab[:,:,2],
                img_resized[:,:,0], img_resized[:,:,1], img_resized[:,:,2]]
    color_stats = []
    for ch in channels:
        mv = ch[mask > 0].astype(np.float32)
        if len(mv) == 0:
            color_stats.extend([0, 0, 0, 0])
        else:
            color_stats.extend([mv.mean(), mv.std(),
                                 float(np.percentile(mv, 25)), float(np.percentile(mv, 75))])
    color_stats = np.array(color_stats) / 255.0
    pixels = img_resized[mask > 0].reshape(-1, 3).astype(np.float32)
    dominant_color_feats = np.zeros(9)
    if len(pixels) >= 3:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(labels.flatten(), minlength=3).astype(np.float32)
        order = np.argsort(-counts)
        dominant_color_feats = (centers[order] / 255.0).flatten()
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    distances = [1, 3, 5]
    glcm = graycomatrix(gray_masked, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    glcm_features = np.concatenate([
        graycoprops(glcm, 'contrast').flatten(), graycoprops(glcm, 'correlation').flatten(),
        graycoprops(glcm, 'energy').flatten(), graycoprops(glcm, 'homogeneity').flatten()
    ])
    lbp = local_binary_pattern(gray_masked, N_POINTS, RADIUS, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), range=(0, N_POINTS + 2))
    hist_lbp = hist_lbp.astype(float)
    hist_lbp /= (hist_lbp.sum() + 1e-7)
    moments_c = cv2.moments(final_contour)
    hu_moments = cv2.HuMoments(moments_c).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    area = cv2.contourArea(final_contour)
    perimeter = cv2.arcLength(final_contour, True)
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-7)
    x, y_bb, w, h_bbox = cv2.boundingRect(final_contour)
    aspect_ratio = w / (h_bbox + 1e-7)
    extent = area / (w * h_bbox + 1e-7)
    hull_area = cv2.contourArea(cv2.convexHull(final_contour))
    solidity = area / (hull_area + 1e-7)
    equiv_diameter = np.sqrt(4 * area / np.pi + 1e-7)
    shape_features = np.array([circularity, aspect_ratio, extent, solidity, equiv_diameter / IMG_SIZE])
    if moments_c["m00"] != 0:
        cx = int(moments_c["m10"] / moments_c["m00"])
        cy = int(moments_c["m01"] / moments_c["m00"])
    else:
        cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    if len(final_contour) > 0:
        pts = final_contour.reshape(-1, 2).astype(np.float32)
        dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
        dists_norm = dists / (dists.max() + 1e-7)
        shape_hist, _ = np.histogram(dists_norm, bins=8, range=(0, 1))
        shape_hist = shape_hist.astype(float)
        shape_hist /= (shape_hist.sum() + 1e-7)
    else:
        shape_hist = np.zeros(8)
    return np.concatenate([color_hist, color_stats, dominant_color_feats,
                           glcm_features, hist_lbp, hu_moments, shape_features, shape_hist])

# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.markdown("### 1. Pilih Jenis Buah")
selected_fruit = st.sidebar.selectbox(
    "Jenis buah yang akan diperiksa:",
    options=["-- Pilih --"] + AVAILABLE_FRUITS,
    index=0
)

st.sidebar.markdown("### 2. Upload Foto")
uploaded_file = st.sidebar.file_uploader(
    "Foto buah (JPG / PNG)",
    type=['jpg', 'jpeg', 'png'],
    disabled=(selected_fruit == "-- Pilih --")
)

if selected_fruit != "-- Pilih --":
    classes = ml_pipeline['disease_classes'].get(selected_fruit, [])
    st.sidebar.markdown("<hr style='border-color:#21262d'>", unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <div style='font-family:monospace; font-size:0.75rem; color:#7d8590; line-height:1.8'>
    <b style='color:#8b949e'>Kondisi yang dideteksi:</b><br>
    {"<br>".join([f"· {c.replace('_', ' ')}" for c in classes])}
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# MAIN
# ==============================================================================
if selected_fruit == "-- Pilih --":
    st.markdown("""
    <div class="select-hint">
        <div style="font-size:2.5rem; margin-bottom:1rem">🌿</div>
        <div>Pilih jenis buah terlebih dahulu di sidebar kiri,<br>kemudian upload foto untuk memulai analisis.</div>
    </div>
    """, unsafe_allow_html=True)

elif uploaded_file is None:
    st.markdown(f"""
    <div class="select-hint">
        <div style="font-size:2.5rem; margin-bottom:1rem">📷</div>
        <div>Buah dipilih: <b style="color:#e6edf3">{selected_fruit}</b><br>
        Silakan upload foto {selected_fruit} di sidebar untuk memulai analisis.</div>
    </div>
    """, unsafe_allow_html=True)

else:
    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:
        img_pil = Image.open(uploaded_file)
        st.image(img_pil, caption=f"Input: {selected_fruit}", use_container_width=True)

    with col2:
        with st.spinner("Menganalisis..."):
            img_array = np.array(img_pil)
            if img_array.ndim == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            if img_array.shape[-1] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_bgr = preprocess_to_studio_style(img_bgr)
            features = extract_advanced_features_segmented(img_bgr)
            features_scaled = ml_pipeline['global_scaler'].transform([features])

        d_mod = ml_pipeline['disease_models'][selected_fruit]['model']
        d_le  = ml_pipeline['le_diseases'][selected_fruit]
        d_type = ml_pipeline['disease_models'][selected_fruit]['type']

        d_probs = d_mod.predict_proba(features_scaled)[0]
        d_idx   = np.argmax(d_probs)
        d_name  = d_le.inverse_transform([d_idx])[0]
        d_conf  = d_probs[d_idx] * 100
        clean_name = d_name.replace("_", " ")

        engine_label = "SVM" if d_type == 'svm' else "XGBoost"
        is_healthy = "Healthy" in clean_name or "healthy" in clean_name
        result_class = "result-healthy" if is_healthy else "result-rotten"
        bar_color = "#3fb950" if is_healthy else "#f85149"
        icon = "✓" if is_healthy else "✗"

        st.markdown(f"""
        <div class="stage-card">
            <div class="stage-label">Buah · {selected_fruit}</div>
            <div class="result-main {result_class}">{icon} {clean_name}</div>
            <div style="color:#7d8590; font-size:0.85rem; margin-top:2px">Confidence {d_conf:.1f}%</div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width:{d_conf}%; background:{bar_color}"></div>
            </div>
            <span class="engine-badge">Spesialis {selected_fruit} · {engine_label}</span>
        </div>
        """, unsafe_allow_html=True)

        # Semua kelas dan probabilitasnya
        st.markdown("""
        <div class="stage-card">
            <div class="stage-label">Distribusi Probabilitas</div>
        """, unsafe_allow_html=True)

        sorted_idx = np.argsort(d_probs)[::-1]
        for i in sorted_idx:
            cls_name = d_le.inverse_transform([i])[0].replace("_", " ")
            pct = d_probs[i] * 100
            is_h = "Healthy" in cls_name or "healthy" in cls_name
            bar_c = "#3fb950" if is_h else "#f85149"
            st.markdown(f"""
            <div class="prob-row">
                <span style="color:#c9d1d9">{cls_name}</span>
                <span style="color:#7d8590">{pct:.1f}%</span>
            </div>
            <div class="confidence-bar-bg" style="margin-bottom:6px">
                <div class="confidence-bar-fill" style="width:{pct}%; background:{bar_c}"></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
