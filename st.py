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
    return cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]

def watercolor_effect(img):
    img_blur = cv2.medianBlur(img, 7)
    edge = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 9, 9)
    img_color = cv2.bilateralFilter(img, 9, 300, 300)
    return cv2.bitwise_and(img_color, img_color, mask=edge)

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
