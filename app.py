from typing import Optional
import streamlit as st
from PIL import Image
from utils import count_parameters, get_model, inference, preprocessing
import numpy as np

import albumentations as A

st.set_page_config(layout="wide")

with st.spinner("Loading model ..."):
    network, sampler = get_model()


st.title("DocDiff: Document Enhancement via Residual Diffusion Models")
st.text(f"Number of parameters: {count_parameters(network)}")

d = dict()
uploaded_file = st.file_uploader("Choose a image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    d["ori_img"] = Image.open(uploaded_file).convert("RGB")

    blur_limit = st.slider("Blur min", min_value=1, max_value=100, step=2)
    blur = A.Blur(blur_limit=blur_limit, always_apply=True, p=1)
    cols = st.columns(2)
    with cols[0]:
        st.header("Original image")
        st.image(d["ori_img"])
    with cols[1]:
        st.header("Blur image")
        ori_img_arr = np.array(d["ori_img"])
        d["blur_image"] = Image.fromarray(blur(image=ori_img_arr)["image"])
        st.image(d["blur_image"])

is_inference = st.button("Inference")
if is_inference and "blur_image" in d:
    img = preprocessing(d["blur_image"])
    with st.spinner("Inferencing ... "):
        noisy_image, init_predict, sampledImgs, finalImgs = inference(
            network, sampler, img
        )

    cols = st.columns(4)
    with cols[0]:
        st.header("Noisy image")
        st.image(noisy_image)

    with cols[1]:
        st.header("Init prediction")
        st.image(init_predict)

    with cols[2]:
        st.header("Sampled imgs")
        st.image(sampledImgs)

    with cols[3]:
        st.header("Final imgs")
        st.image(finalImgs)
