import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

st.set_page_config(
    page_title="Reconocimiento de Imagen con Isa",
    page_icon="📸",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins&display=swap');

html, body, .stApp {
    background: linear-gradient(to bottom right, #a18cd1, #fbc2eb);
    color: #2d2d2d;
    font-family: 'Poppins', sans-serif;
    text-align: center;
}

h1, h2, h3, h4, h5, h6, .stTitle, .stHeader {
    color: #ff5e7e;
    text-align: center;
}

.stImage > img {
    display: block;
    margin-left: auto;
    margin-right: auto;
}

.stButton>button {
    background-color: #ff5e7e;
    color: white;
    font-weight: bold;
    border-radius: 10px;
}

.block-container {
    padding-left: 5%;
    padding-right: 5%;
}

.stSidebar > div:first-child {
    background-color: #fff3f8;
    color: #2d2d2d;
    font-family: 'Poppins', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("✨ ¿Estás con Isa? ✨")
st.write("Sabremos si estás con Isa o no", platform.python_version())

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image = Image.open('Isa.png')
st.image(image, width=350)

with st.sidebar:
    st.subheader("🌈 Usa tu modelo de Teachable Machine y mira si estás con Isa")
    st.write("Carga o toma una imagen y veremos si en tu imagen estás con Isa o no💫")

img_file_buffer = st.camera_input("📷 ¡Sonríe y toma una foto!")

if img_file_buffer is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)
    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)

    if prediction[0][0]>0.5:
        st.header('🎉 ¡Con Isa detectada! Probabilidad: ' + str(prediction[0][0]))
    if prediction[0][1]>0.5:
        st.header('🤖 ¡Sin Isa detectada! Probabilidad: ' + str(prediction[0][1]))


