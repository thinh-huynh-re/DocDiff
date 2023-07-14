import sys
from typing import Tuple

from PIL import Image
import torch
from torchvision.transforms import ToTensor, RandomCrop, ToPILImage
from torchvision.utils import save_image

from model.DocDiff import DocDiff
from schedule.diffusionSample import GaussianDiffusion
from schedule.schedule import Schedule

import numpy as np
import streamlit as st

from torch import Tensor, nn

TEST_INITIAL_PREDICTOR_WEIGHT_PATH = "./checksave/init.pth"
TEST_DENOISER_WEIGHT_PATH = "./checksave/denoiser.pth"

# image pre-processing
seed = torch.random.seed()
torch.random.manual_seed(seed)


def min_max(array: Tensor) -> Tensor:
    return (array - array.min()) / (array.max() - array.min())


def crop_concat(img: np.ndarray, size=128):
    shape = img.shape
    correct_shape = (size * (shape[2] // size + 1), size * (shape[3] // size + 1))
    one = torch.ones((shape[0], shape[1], correct_shape[0], correct_shape[1]))
    one[:, :, : shape[2], : shape[3]] = img
    # crop
    for i in range(shape[2] // size + 1):
        for j in range(shape[3] // size + 1):
            if i == 0 and j == 0:
                crop = one[:, :, i * size : (i + 1) * size, j * size : (j + 1) * size]
            else:
                crop = torch.cat(
                    (
                        crop,
                        one[:, :, i * size : (i + 1) * size, j * size : (j + 1) * size],
                    ),
                    dim=0,
                )
    return crop


def crop_concat_back(img: np.ndarray, prediction: np.ndarray, size=128) -> np.ndarray:
    shape = img.shape
    for i in range(shape[2] // size + 1):
        for j in range(shape[3] // size + 1):
            if j == 0:
                crop = prediction[
                    (i * (shape[3] // size + 1) + j)
                    * shape[0] : (i * (shape[3] // size + 1) + j + 1)
                    * shape[0],
                    :,
                    :,
                    :,
                ]
            else:
                crop = torch.cat(
                    (
                        crop,
                        prediction[
                            (i * (shape[3] // size + 1) + j)
                            * shape[0] : (i * (shape[3] // size + 1) + j + 1)
                            * shape[0],
                            :,
                            :,
                            :,
                        ],
                    ),
                    dim=3,
                )
        if i == 0:
            crop_concat = crop
        else:
            crop_concat = torch.cat((crop_concat, crop), dim=2)
    return crop_concat[:, :, : shape[2], : shape[3]]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_format = "{:,}".format


def preprocessing(img: Image.Image) -> Tensor:
    img = RandomCrop(304, pad_if_needed=True, padding_mode="reflect")(img)
    img = ToTensor()(img)  # convert to tensor
    return img.unsqueeze(0)  # add batch dimension (1, 3, 300, 300)


def postprocessing(img: Tensor) -> Image.Image:
    return ToPILImage()(img)


def count_parameters(model: nn.Module) -> str:
    """Count the number of learnable parameters of a model"""
    return num_format(sum(p.numel() for p in model.parameters() if p.requires_grad))


@st.cache_resource
def get_model():
    network = DocDiff(
        input_channels=3 + 3,
        output_channels=3,
        n_channels=32,
        ch_mults=[1, 2, 3, 4],
        n_blocks=1,
    ).to(
        device
    )  # initialize network

    network.init_predictor.load_state_dict(
        torch.load(TEST_INITIAL_PREDICTOR_WEIGHT_PATH, map_location=device)
    )  # load weights
    network.denoiser.load_state_dict(
        torch.load(TEST_DENOISER_WEIGHT_PATH, map_location=device)
    )

    network.eval()
    schedule = Schedule("linear", 100)  # initialize schedule
    sampler = GaussianDiffusion(network.denoiser, 100, schedule).to(
        device
    )  # initialize diffusion sampler
    return network, sampler


@torch.no_grad()
def inference(
    network: DocDiff, sampler: GaussianDiffusion, img: Tensor
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    noisyImage = torch.randn_like(img).to(device)
    init_predict: Tensor = network.init_predictor(img.to(device), 0)
    sampledImgs: Tensor = sampler(
        noisyImage.to(device), init_predict, "True"
    )  # full-size sampling
    finalImgs = sampledImgs + init_predict

    return (
        postprocessing(noisyImage.cpu().squeeze()),
        postprocessing(init_predict.cpu().squeeze()),
        postprocessing(min_max(sampledImgs.cpu()).squeeze()),
        postprocessing(finalImgs.cpu().squeeze()),
    )
