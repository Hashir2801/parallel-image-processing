# ============================================================
# Parallel Image Processing System Using OpenCV (With UI)
# UI: Streamlit (User uploads images + sees comparison + graph)
# ============================================================

import time
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# ------------------------------------------------------------
# 1) Helper: Convert uploaded file -> OpenCV image (BGR)
# ------------------------------------------------------------
def uploaded_file_to_bgr(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return bgr

# ------------------------------------------------------------
# 2) Image Filters
# ------------------------------------------------------------
def apply_blur(bgr: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(bgr, (15, 15), 0)

def apply_sharpen(bgr: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0],
                       [-1,  5, -1],
                       [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(bgr, -1, kernel)

def process_one_image(bgr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "Blur":
        return apply_blur(bgr)
    if mode == "Sharpen":
        return apply_sharpen(bgr)
    return apply_sharpen(apply_blur(bgr))  # Both

# ------------------------------------------------------------
# 3) Sequential Processing
# ------------------------------------------------------------
def run_sequential(images_bgr: list[np.ndarray], mode: str):
    outputs = []
    start = time.perf_counter()
    for img in images_bgr:
        outputs.append(process_one_image(img, mode))
    end = time.perf_counter()
    return outputs, end - start

# ------------------------------------------------------------
# 4) Parallel Processing
# ------------------------------------------------------------
def run_parallel(images_bgr: list[np.ndarray], mode: str):
    start = time.perf_counter()
    with ThreadPoolExecutor() as ex:
        outputs = list(ex.map(lambda im: process_one_image(im, mode), images_bgr))
    end = time.perf_counter()
    return outputs, end - start

# ------------------------------------------------------------
# 5) UI (Streamlit)
# ------------------------------------------------------------
st.set_page_config(page_title="Parallel Image Processing (OpenCV)", layout="wide")

st.title("Parallel Image Processing System Using OpenCV")
st.write("Upload images, choose a filter, and compare Sequential vs Parallel processing time.")

uploaded_files = st.file_uploader(
    "Upload one or more images (jpg/png)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

mode = st.selectbox("Choose Filter", ["Blur", "Sharpen", "Both (Blur + Sharpen)"])

# Columns for images (output preview)
colA, colB = st.columns(2)

run_btn = st.button("Run Comparison")

if run_btn:
    if not uploaded_files or len(uploaded_files) == 0:
        st.error("Please upload at least 1 image.")
        st.stop()

    images_bgr = []
    names = []
    for f in uploaded_files:
        bgr = uploaded_file_to_bgr(f)
        if bgr is None:
            st.warning(f"Could not read: {f.name}")
            continue
        images_bgr.append(bgr)
        names.append(f.name)

    if len(images_bgr) == 0:
        st.error("No valid images found. Please upload valid jpg/png files.")
        st.stop()

    # Run sequential + parallel
    seq_outputs, seq_time = run_sequential(images_bgr, mode)
    par_outputs, par_time = run_parallel(images_bgr, mode)

    # ----- Time Comparison (Metrics) -----
    st.subheader("Time Comparison")
    c1, c2, c3 = st.columns(3)
    c1.metric("Images", str(len(images_bgr)))
    c2.metric("Sequential Time (seconds)", f"{seq_time:.4f}")
    c3.metric("Parallel Time (seconds)", f"{par_time:.4f}")

    # ----- Graph (Place it here, under filter + time comparison) -----
    st.subheader("Graph Comparison (Sequential vs Parallel)")

    labels = ["Sequential", "Parallel"]
    times = [seq_time, par_time]

        # ----- Output Preview (Moved UP: right after filter + graph) -----
    with colA:
        st.markdown("### Sequential Output")
        for name, out_bgr in zip(names, seq_outputs):
            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            st.image(out_rgb, caption=f"{name} (Sequential)", use_container_width=True)

    with colB:
        st.markdown("### Parallel Output")
        for name, out_bgr in zip(names, par_outputs):
            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            st.image(out_rgb, caption=f"{name} (Parallel)", use_container_width=True)

    fig, ax = plt.subplots()
    ax.bar(labels, times)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Execution Time Comparison")
    st.pyplot(fig)

# ============================================================END ============================================================