import streamlit as st
import numpy as np
import os
from PIL import Image
from utils import get_model, preprocessing, inference

st.set_page_config(layout="wide")

with st.spinner("Loading model ..."):
    network, sampler = get_model()

st.title("DocDiff: Document Enhancement via Residual Diffusion Models")

uploaded_file = st.file_uploader("Choose a image file", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    # Convert the file to an opencv image.
    ori_img = Image.open(uploaded_file).convert("RGB")

    img = preprocessing(ori_img)
    with st.spinner("Inferencing ... "):
        noisy_image, init_predict, sampledImgs, finalImgs = inference(
            network, sampler, img
        )

    cols = st.columns(5)
    with cols[0]:
        st.header("Original img")
        st.image(ori_img)

    with cols[1]:
        st.header("Init prediction")
        st.image(init_predict)

    with cols[2]:
        st.header("Sampled imgs")
        st.image(sampledImgs)

    with cols[3]:
        st.header("Final imgs")
        st.image(finalImgs)

    with cols[4]:
        st.header("Noisy image")
        st.image(noisy_image)
