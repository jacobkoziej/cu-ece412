# SPDX-License-Identifier: GPL-3.0-or-later
#
# freakiir.py -- freakIIR: IIRNet but with phase
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>
# Copyright (C) 2024  Eric Eng <eric.eng@cooper.edu>

import numpy as np
import torch

from math import pi

from einops import (
    rearrange,
    reduce,
)
from pytorch_lightning import LightningModule
from torch import (
    nn,
    optim,
)
from torch.utils.data import Dataset

from dsp import freqz_zpk


class FreakIir(LightningModule):
    def __init__(
        self,
        inputs: int = 1024,
        layers: int = 16,
        hidden_dimension: int = 2 * 1024,
        sections: int = 2,
        *,
        negative_slope: float = 0.2,
        alpha: float = 0.5,
    ):
        assert not inputs % 2
        assert layers >= 2

        super().__init__()

        self.save_hyperparameters()

        self.N = inputs // 2

        self.layers = nn.ModuleList()

        layers -= 2

        def gen_layer(in_dimension, out_dimension):
            return nn.Sequential(
                nn.Linear(in_dimension, out_dimension),
                nn.LayerNorm(out_dimension),
                nn.LeakyReLU(negative_slope),
            )

        self.layers.append(gen_layer(inputs, hidden_dimension))

        for layer in range(layers):
            self.layers.append(gen_layer(hidden_dimension, hidden_dimension))

        self.layers.append(nn.Linear(hidden_dimension, sections * 8))

        self.mse_loss = nn.MSELoss()
        self.loss = nn.MSELoss()

    def _step(self, batch, batch_idx, log):
        riemann_sphere, spectrum, cepstrum, input = batch

        prediction = self.forward(input)
        prediction = output2riemann_sphere(prediction, self.hparams.sections)

        loss = self.loss(prediction, riemann_sphere)

        self.log(f"{log}/loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=1e-6,
        )

        return optimizer

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)

        return x

    def magnitude_loss(
        self, prediction: torch.Tensor, spectrum: torch.Tensor, cepstrum: torch.Tensor
    ):
        h = riemann_sphere2dft(prediction, self.N)
        c = dft2cepstrum(h)

        alpha = self.hparams.alpha
        mse_loss = self.mse_loss

        loss = torch.tensor([0.0]).to(prediction.device)

        loss += (1 - alpha) * mse_loss(
            torch.log10(h.abs()), torch.log10(spectrum.abs())
        )
        loss += alpha * mse_loss(c.abs() ** 2, cepstrum.abs() ** 2)

        return loss

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")


class FreakIirDataset(Dataset):
    def __init__(self, dataset: torch.Tensor, sections: int, *, N: int = 512):
        self.riemann_sphere = rearrange(
            dataset, "(batch sections) h -> batch sections h", sections=sections
        )

        h = riemann_sphere2dft(self.riemann_sphere, N)

        self.spectrum = h
        self.cepstrum = dft2cepstrum(h)

        self.input = dft2input(h)

    def __len__(self):
        return self.riemann_sphere.shape[0]

    def __getitem__(self, item):
        riemann_sphere = self.riemann_sphere[item]
        spectrum = self.spectrum[item]
        cepstrum = self.cepstrum[item]
        input = self.input[item]

        return riemann_sphere, spectrum, cepstrum, input


def dft2input(f: torch.Tensor):
    z = torch.stack([20 * torch.log10(f.abs()), f.angle()], axis=-1)

    return rearrange(z, "... w z -> ... (w z)")


def dft2cepstrum(f: torch.Tensor) -> torch.Tensor:
    z = torch.log(f)

    return torch.fft.ifft(z)


def output2riemann_sphere(o: torch.Tensor, sections: int):
    return rearrange(
        o, "... batch (sections h) -> ... batch sections h", sections=sections
    )


def riemann_sphere2dft(r: torch.Tensor, N: int) -> torch.Tensor:
    zp = torch.exp(1j * r[..., ::2]) * (1 / torch.tan(0.5 * r[..., 1::2]))

    z = zp[..., :2]
    p = zp[..., 2:]

    _, h = freqz_zpk(z, p, 1, N)

    h = reduce(h, "... sections h -> ... h", "prod")

    return h
