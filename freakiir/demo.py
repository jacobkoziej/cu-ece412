# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# SPDX-License-Identifier: GPL-3.0-or-later
#
# demo.py -- let's get freakIIR
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>
# Copyright (C) 2024  Eric Eng <eric.eng@cooper.edu>

# %% [markdown]
# # Setup

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from math import pi

from einops import (
    rearrange,
    reduce,
)
from torch.fft import fftshift
from scipy.signal import lfilter

from freakiir import (
    FreakIir,
    FreakIirDataset,
    freqz_zpk,
    riemann_sphere2dft,
)

# %%
model = FreakIir.load_from_checkpoint("checkpoints/ideal.ckpt")
model.eval()

# %%
order = 4
sections = order // 2

# %%
dataset_root = os.path.join(
    os.environ.get("DATASETS_PATH", "."), f"freakIIR/{order}"
)
test = FreakIirDataset(
    torch.tensor(
        pd.read_csv(os.path.join(dataset_root, "test.csv")).values,
        dtype=torch.float32,
    ),
    sections,
)

# %% [markdown]
# ## Data Pipeline

# %%
riemann_sphere = test[0][0]

# %% tags=["active-ipynb"]
riemann_sphere

# %% [markdown]
# ### Filter Response
#
# We get project our poles and zeros off of the Riemann Sphere onto the
# complex plane so that we can get the frequency response of the
# sections.

# %%
r = riemann_sphere
zp = torch.exp(1j * r[..., ::2]) * (1 / torch.tan(0.5 * r[..., 1::2]))

z = zp[..., :2]
p = zp[..., 2:]

# %% tags=["active-ipynb"]
z

# %% tags=["active-ipynb"]
p

# %%
N = model.hparams.inputs // 2

w, h = freqz_zpk(z, p, 1, N)
h = reduce(h, "... sections h -> ... h", "prod")
h = fftshift(h)


# %%
def plot_db_mag(h):
    ax = plt.subplot()

    N = len(h)

    ax.plot(np.linspace(-pi, pi, N), 20 * np.log10(np.abs(h)))
    ax.set_xticks(
        [-pi, -pi / 2, 0, pi / 2, pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )
    ax.set_xlabel(r"$\omega$ [rad]")
    ax.set_ylabel("$|H|$ [dB]")


# %%
def plot_phase(h):
    ax = plt.subplot()

    N = len(h)

    ax.plot(np.linspace(-pi, pi, N), np.unwrap(np.rad2deg(np.angle(h))))
    ax.set_xticks(
        [-pi, -pi / 2, 0, pi / 2, pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )
    ax.set_xlabel(r"$\omega$ [rad]")
    ax.set_ylabel("Angle [deg]")


# %% tags=["active-ipynb"]
plot_db_mag(h.numpy())

# %% tags=["active-ipynb"]
plot_phase(h.numpy())

# %% [markdown]
# # Evaluation
#
# Since we can't feed complex values into an MLP (yet), we'll instead
# feed in the dB magnitude response along with the phase.

# %%
input = torch.stack([20 * torch.log10(h.abs()), h.angle()], axis=-1)
input = rearrange(input, "... w z -> ... (w z)")

# %%
prediction = model.forward(torch.unsqueeze(input, 0))

# %% [markdown]
# Since our model returns a flattened vector, we must rearrange the
# output to be in terms of sections.

# %%
prediction = (
    rearrange(
        prediction,
        "... batch (sections h) -> ... batch sections h",
        sections=model.hparams.sections,
    )
    .squeeze()
    .detach()
)

# %% tags=["active-ipynb"]
prediction

# %%
h = riemann_sphere2dft(prediction, N)
h = fftshift(h).numpy()

# %% tags=["active-ipynb"]
plot_db_mag(h)

# %% tags=["active-ipynb"]
plot_phase(h)

# %% [markdown]
# # Results


# %%
def ax_stem(ax, log, type, name, linefmt):

    ax.stem(log["step"], log["value"], label=f"{type}/{name}", linefmt=linefmt)


# %%
def lpf_plot(ax, log, type, name, color):
    alpha = 0.6

    value = lfilter([1 - alpha], [1, -alpha], log["value"])

    ax.plot(log["step"], value, label=f"{type}/{name}", color=color)


# %%
ax = plt.subplot()

log = pd.read_csv("logs/val/loss.csv")
ax_stem(ax, log, "val", "loss", "r--")

log = pd.read_csv("logs/train/loss.csv")
lpf_plot(ax, log, "train", "loss", "blue")

_ = ax.set_title("Loss Rates")
_ = ax.set_xlabel("Step")
_ = ax.set_ylabel("Loss")
_ = ax.legend()

# %% [markdown]
# ```
# test/loss    1.7101298570632935
# ```
# chat we're cooked 💀
