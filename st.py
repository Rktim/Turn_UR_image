import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def sketch_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    return cv2.Canny(blur, 50, 150)

def binary_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def watercolor_effect(img):
    img_blur = cv2.medianBlur(img, 7)
    img_color = cv2.edgePreservingFilter(img_blur, flags=1, sigma_s=60, sigma_r=0.4)
    img_cartoon = cv2.stylization(img_color, sigma_s=150, sigma_r=0.25)
    return img_cartoon

st.title("Turn UR Image 🖼️")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if "effect" not in st.session_state:
    st.session_state.effect = None

if uploaded_file:
    image = Image.open(uploaded_file)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    col1, col2, col3 = st.columns(3)
    
    if col1.button("Sketch"):
        st.session_state.effect = "Sketch"
    if col2.button("Black N White"):
        st.session_state.effect = "Binary"
    if col3.button("Water Painting"):
        st.session_state.effect = "Watercolor"
    
    result = None
    if st.session_state.effect == "Sketch":
        result = sketch_effect(img_bgr)
    elif st.session_state.effect == "Binary":
        result = binary_effect(img_bgr)
    elif st.session_state.effect == "Watercolor":
        result = watercolor_effect(img_bgr)
    
    if result is not None:
        st.image(result, caption=f"{st.session_state.effect} Effect", use_container_width=True)
        result_pil = Image.fromarray(result) if result.ndim == 2 else Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(label="Download Image", data=byte_im, file_name="transformed_image.png", mime="image/png")
