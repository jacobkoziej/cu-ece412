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
import matplotlib.pyplot as plt
import numpy as np
import torch

from einops import repeat

from dataset import RandomDataset
from freakiir import FreakIir
from generate import (
    uniform_half_disk,
    uniform_half_ring,
)

from torch import pi


# %%
def plot_db_mag(h):
    ax = plt.subplot()

    N = h.shape[-1]

    theta = np.linspace(0, pi, N)

    if h.ndim > 1:
        theta = repeat(theta, "theta -> plots theta", plots=h.shape[0])

    ax.plot(theta.T, 20 * np.log10(np.abs(h.T)))
    ax.set_xticks(
        [0, pi / 4, pi / 2, 3 * pi / 4, pi],
        ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"],
    )
    ax.set_xlabel(r"$\omega$ [rad]")
    ax.set_ylabel("$|H|$ [dB]")
    ax.set_title("Magnitude Response")


# %%
def plot_unwrapped_phase(h):
    ax = plt.subplot()

    N = h.shape[-1]

    theta = np.linspace(0, pi, N)

    if h.ndim > 1:
        theta = repeat(theta, "theta -> plots theta", plots=h.shape[0])

    ax.plot(theta.T, np.unwrap(np.rad2deg(np.angle(h))).T)
    ax.set_xticks(
        [0, pi / 4, pi / 2, 3 * pi / 4, pi],
        ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"],
    )
    ax.set_xlabel(r"$\omega$ [rad]")
    ax.set_ylabel("Angle [deg]")
    ax.set_title("Phase Response")


# %%
def plot_zplane(z, p):
    ax = plt.subplot()

    theta = np.linspace(0, 2 * pi, 1_000)

    ax.plot(np.cos(theta), np.sin(theta), "k--")
    ax.plot(np.real(p), np.imag(p), "rx", markersize=10)
    ax.plot(np.real(z), np.imag(z), "bo", markersize=10, fillstyle="none")

    ax.set_xlabel("Re($z$)")
    ax.set_ylabel("Im($z$)")
    ax.set_title("Pole-Zero Plot")
    ax.set_aspect("equal")
    ax.grid(True)


# %%
order = 8
sections = order // 2

# %%
model_min_phase = FreakIir.load_from_checkpoint(
    f"ckpt/{order}/min-phase/checkpoints/ideal.ckpt"
)
model_min_phase = model_min_phase.eval()

# %% tags=["active-ipynb"]
model_min_phase

model_all_pass = FreakIir.load_from_checkpoint(
    f"ckpt/{order}/all-pass/checkpoints/ideal.ckpt"
)
model_all_pass = model_all_pass.eval()

# %% tags=["active-ipynb"]
model_all_pass

# %%
dataset_min_phase = RandomDataset(
    generator=uniform_half_disk,
    sections=sections,
    all_pass=False,
)

# %%
dataset_all_pass = RandomDataset(
    generator=uniform_half_ring,
    sections=sections,
    all_pass=True,
)

# %% [markdown]
# ## Minimum Phase

# %%
zp_min_phase, h_min_phase = dataset_min_phase[0]
z_min_phase = zp_min_phase[..., 0, :].flatten()
p_min_phase = zp_min_phase[..., 1, :].flatten()

# %% tags=["active-ipynb"]
plot_zplane(z_min_phase.numpy(), p_min_phase.numpy())

# %% tags=["active-ipynb"]
plot_db_mag(h_min_phase.numpy())

# %% [markdown]
# ## All-Pass

# %%
zp_all_pass, h_all_pass = dataset_all_pass[0]
z_all_pass = zp_all_pass[..., 0, :].flatten()
p_all_pass = zp_all_pass[..., 1, :].flatten()

# %% tags=["active-ipynb"]
plot_zplane(z_all_pass.numpy(), p_all_pass.numpy())

# %% tags=["active-ipynb"]
plot_unwrapped_phase(h_all_pass.numpy())

# %% [markdown]
# # Evaluation

# %% [markdown]
# ## Minimum Phase

# %%
prediction_zp_min_phase = (
    model_min_phase.forward(
        model_min_phase._dft2input(h_min_phase.unsqueeze(0))
    )
    .detach()
    .squeeze()
)
prediction_z_min_phase = prediction_zp_min_phase[..., 0, :]
prediction_p_min_phase = prediction_zp_min_phase[..., 1, :]

# %% tags=["active-ipynb"]
plot_zplane(prediction_z_min_phase.numpy(), p_min_phase.numpy())

# %%
prediction_h_min_phase = model_min_phase._zp2dft(prediction_zp_min_phase)
h_min_phase = torch.cat(
    [h_min_phase.unsqueeze(0), prediction_h_min_phase.unsqueeze(0)]
)

# %% tags=["active-ipynb"]
plot_db_mag(h_min_phase.numpy())

# %% [markdown]
# ## All-Pass

# %%
prediction_zp_all_pass = (
    model_all_pass.forward(model_all_pass._dft2input(h_all_pass.unsqueeze(0)))
    .detach()
    .squeeze()
)
prediction_z_all_pass = prediction_zp_all_pass[..., 0, :]
prediction_p_all_pass = prediction_zp_all_pass[..., 1, :]

# %% tags=["active-ipynb"]
plot_zplane(prediction_z_all_pass.numpy(), p_all_pass.numpy())

# %%
prediction_h_all_pass = model_all_pass._zp2dft(prediction_zp_all_pass)
h_all_pass = torch.cat(
    [h_all_pass.unsqueeze(0), prediction_h_all_pass.unsqueeze(0)]
)

# %% tags=["active-ipynb"]
plot_unwrapped_phase(h_all_pass.numpy())
