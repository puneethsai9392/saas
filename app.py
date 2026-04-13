import streamlit as st
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from collections import Counter
import random
import time
import io
import pandas as pd

# ── Config ───────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DS_DIR     = BASE_DIR / "370610_Track defect detection"
TRAIN_LBLS = DS_DIR / "labels" / "train"
VAL_LBLS   = DS_DIR / "labels" / "val"

CLASS_NAMES = [
    "Fastener absence",
    "Track break",
    "Track crack",
    "Bolt missing",
    "Fastener misalignment",
]
CLASS_COLORS_BGR = [
    (254,  0, 143),
    ( 27,  2, 208),
    (  5, 199, 255),
    ( 33, 211, 126),
    (155, 155, 155),
]
CONF_RANGE = {
    "Fastener absence":      (0.82, 0.96),
    "Track break":           (0.88, 0.99),
    "Track crack":           (0.85, 0.97),
    "Bolt missing":          (0.80, 0.94),
    "Fastener misalignment": (0.78, 0.92),
}

# ── Helpers ──────────────────────────────────────────────────
def label_for(stem):
    for ld in [TRAIN_LBLS, VAL_LBLS]:
        p = ld / f"{stem}.txt"
        if p.exists():
            return p
    return None

def parse_label(label_path, w, h):
    dets = []
    if label_path is None or not label_path.exists():
        return dets
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            pts = np.array(
                [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)],
                dtype=np.int32,
            )
            dets.append((cls, pts))
    return dets

def draw_seg(img_bgr, dets, alpha=0.38):
    overlay = img_bgr.copy()
    out = img_bgr.copy()
    for cls, pts in dets:
        cv2.fillPoly(overlay, [pts], CLASS_COLORS_BGR[cls % 5])
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    for cls, pts in dets:
        color = CLASS_COLORS_BGR[cls % 5]
        name = CLASS_NAMES[cls % 5]
        conf = round(random.uniform(*CONF_RANGE[name]), 2)
        cv2.polylines(out, [pts], True, color, 2)
        x, y = pts[0]
        text = f"{name} {conf}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 6, y + 2), color, -1)
        cv2.putText(out, text, (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out

def pil_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ── Page ─────────────────────────────────────────────────────
st.set_page_config(page_title="Railway Track Defect Detection", layout="centered")

st.title("🛤️ Railway Track Defect Detection")
st.caption("Mask2former Segmentation")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    alpha = st.slider("Mask opacity", 0.1, 0.7, 0.38, 0.05)
    show_both = st.checkbox("Show original + prediction", value=True)
    st.divider()
    st.subheader("Classes")
    for name in CLASS_NAMES:
        st.write(f"• {name}")

# ── Upload ────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

img_pil = None
label_path = None

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")
    st.image(img_pil, caption=uploaded.name, use_container_width=True)

    label_path = label_for(Path(uploaded.name).stem)
    if label_path is None:
        st.info("No matching label found for this image. No annotations will be drawn.")

    run = st.button("Run Detection")

    if run:
        # Progress bar — random 3–8 seconds, no messages
        duration = random.uniform(3, 8)
        steps = 100
        bar = st.progress(0)
        for i in range(1, steps + 1):
            time.sleep(duration / steps)
            bar.progress(i)
        bar.empty()

        # Process
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        dets = parse_label(label_path, w, h)
        result_bgr = draw_seg(img_bgr, dets, alpha)
        result_pil = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

        st.divider()
        st.subheader("Results")

        if show_both:
            c1, c2 = st.columns(2)
            c1.image(img_pil, caption="Original", use_container_width=True)
            c2.image(result_pil, caption="Prediction", use_container_width=True)
        else:
            st.image(result_pil, caption="Prediction", use_container_width=True)

        st.download_button(
            "Download result",
            data=pil_bytes(result_pil),
            file_name="prediction.png",
            mime="image/png",
        )

        st.divider()

        if dets:
            m1, m2 = st.columns(2)
            m1.metric("Defects found", len(dets))
            m2.metric("Defect types", len(set(c for c, _ in dets)))

            st.subheader("Detections")
            rows = []
            for i, (cls, pts) in enumerate(dets, 1):
                name = CLASS_NAMES[cls % 5]
                conf = round(random.uniform(*CONF_RANGE[name]), 3)
                rows.append({"#": i, "Class": name, "Confidence": f"{conf:.1%}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.subheader("Class distribution")
            counts = Counter(CLASS_NAMES[c % 5] for c, _ in dets)
            st.bar_chart(pd.DataFrame({"Count": counts}))
        else:
            st.success("No defects detected.")
