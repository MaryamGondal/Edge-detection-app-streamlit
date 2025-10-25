


import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io


# Streamlit Page Configuration

st.set_page_config(page_title="Edge Detection Visualizer", layout="wide")


# Helper Functions

def to_cv2_image(pil_image):
    """Convert PIL image to OpenCV BGR image."""
    rgb = np.array(pil_image)
    if rgb.shape[-1] == 4:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def to_pil_image(cv2_bgr):
    """Convert OpenCV BGR image to PIL image."""
    rgb = cv2.cvtColor(cv2_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def ensure_odd(x):
    """Make sure kernel sizes are odd numbers."""
    x = int(max(1, round(x)))
    if x % 2 == 0:
        x += 1
    return x

def resize_for_display(pil_image, max_width=800, max_height=800):
    """Resize large images for better display performance."""
    w, h = pil_image.size
    scale = min(1.0, max_width / w, max_height / h)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        return pil_image.resize(new_size, Image.ANTIALIAS)
    return pil_image


# Sidebar: Controls

st.sidebar.title("Controls")

uploaded = st.sidebar.file_uploader(
    "Upload an image (JPG, PNG, BMP)", type=["jpg", "jpeg", "png", "bmp"]
)
st.sidebar.markdown("---")

algorithm = st.sidebar.selectbox(
    "Select edge detection algorithm", ("Canny", "Sobel", "Laplacian")
)

grayscale_mode = st.sidebar.checkbox(
    "Convert to grayscale before processing (recommended)", value=True
)

st.sidebar.markdown("### Algorithm parameters")

# Canny parameters
if algorithm == "Canny":
    canny_lower = st.sidebar.slider("Lower threshold", 0, 255, 50)
    canny_upper = st.sidebar.slider("Upper threshold", 0, 255, 150)
    gauss_ksize = st.sidebar.slider("Gaussian kernel size (odd)", 1, 11, 3)
    gauss_sigma = st.sidebar.slider("Gaussian sigma", 0.0, 10.0, 1.0, step=0.1)
    gauss_ksize = ensure_odd(gauss_ksize)

# Sobel parameters
if algorithm == "Sobel":
    sobel_ksize = st.sidebar.slider("Sobel kernel size (odd):", 1, 31, 3)
    sobel_ksize = ensure_odd(sobel_ksize)
    sobel_dir = st.sidebar.radio("Gradient direction", ("Both", "X", "Y"))

# Laplacian parameters
if algorithm == "Laplacian":
    lap_ksize = st.sidebar.slider("Laplacian kernel size (odd)", 1, 31, 3)
    lap_ksize = ensure_odd(lap_ksize)

st.sidebar.markdown("---")
st.sidebar.markdown(
    " Adjust parameters to see changes in real-time."
)


col1, col2 = st.columns([1, 1])
col1.subheader("Input (Original)")
col2.subheader("Output (Edge-detected)")

if uploaded is None:
    with col1:
        st.info("⬅️ Upload an image from the sidebar to get started.")
else:
    # Load uploaded image
    image_data = uploaded.read()
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    display_in = resize_for_display(pil_image, max_width=600, max_height=600)
    with col1:
        st.image(display_in, use_column_width=True, caption="Original Image")

    # Convert to OpenCV BGR
    img_bgr = to_cv2_image(pil_image)

    # Convert to grayscale if chosen
    if grayscale_mode:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        proc_img = gray.copy()
    else:
        proc_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    
    # Algorithm Implementations
    
    if algorithm == "Canny":
        # Gaussian smoothing
        if gauss_ksize > 1:
            proc_img = cv2.GaussianBlur(proc_img, (gauss_ksize, gauss_ksize), sigmaX=gauss_sigma)
        edges = cv2.Canny(proc_img, canny_lower, canny_upper)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        out_pil = to_pil_image(edges_bgr)

    elif algorithm == "Sobel":
        ddepth = cv2.CV_64F
        grad_x = cv2.Sobel(proc_img, ddepth, 1, 0, ksize=sobel_ksize)
        grad_y = cv2.Sobel(proc_img, ddepth, 0, 1, ksize=sobel_ksize)
        if sobel_dir == "X":
            grad = np.absolute(grad_x)
        elif sobel_dir == "Y":
            grad = np.absolute(grad_y)
        else:
            grad = np.hypot(grad_x, grad_y)
        grad = np.uint8(255 * (grad / (grad.max() + 1e-8)))
        edges_bgr = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
        out_pil = to_pil_image(edges_bgr)

    elif algorithm == "Laplacian":
        ddepth = cv2.CV_64F
        lap = cv2.Laplacian(proc_img, ddepth, ksize=lap_ksize)
        lap = np.absolute(lap)
        lap = np.uint8(255 * (lap / (lap.max() + 1e-8)))
        edges_bgr = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
        out_pil = to_pil_image(edges_bgr)

    
    # Display Output and Download Option
    
    display_out = resize_for_display(out_pil, max_width=600, max_height=600)
    with col2:
        st.image(display_out, use_column_width=True, caption=f"Output — {algorithm}")

    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    buf.seek(0)
    st.download_button(
        "📥 Download Result (PNG)",
        data=buf,
        file_name=f"edge_output_{algorithm.lower()}.png",
        mime="image/png",
    )


# Footer: Info Section

st.markdown("---")
st.markdown("### About this Application")
st.markdown("""
This interactive tool demonstrates how **edge detection algorithms** behave under different parameters.

**Algorithms implemented:**
- **Canny** — Multi-stage edge detector with noise reduction.
- **Sobel** — Gradient-based edge detection (X, Y, or both).
- **Laplacian** — Second-derivative edge detector emphasizing rapid intensity changes.

**How to use:**
1. Upload an image.
2. Choose an algorithm from the sidebar.
3. Adjust the parameters and watch the output update in real-time.


""")
