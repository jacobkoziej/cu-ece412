# SPDX-License-Identifier: GPL-3.0-or-later
#
# freakiir.py -- freakIIR: IIRNet but with phase
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>
# Copyright (C) 2024  Eric Eng <eric.eng@cooper.edu>

import numpy as np
import torch

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


class FreakIir(LightningModule):
    def __init__(
        self,
        inputs: int = 1024,
        layers: int = 4,
        hidden_dimension: int = 4 * 1024,
        sections: int = 2,
        *,
        negative_slope: float = 0.2,
    ):
        assert layers >= 2

        super().__init__()

        self.save_hyperparameters()

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
            lr=1e-4,
        )

        return optimizer

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)

        return x

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
    z = torch.stack([10 * torch.log10(f.abs()), f.angle()], axis=-1)

    return rearrange(z, "... w z -> ... (w z)")


def dft2cepstrum(f: torch.Tensor) -> torch.Tensor:
    z = torch.log(f)

    return torch.fft.ifft(z)


def freqz_zpk(
    z: torch.Tensor,
    p: torch.Tensor,
    k: torch.Tensor,
    N: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    from math import pi

    assert z.dtype == p.dtype

    dtype = z.dtype

    w = torch.linspace(0, 2 * pi, N)
    h = torch.exp(1j * w)

    h = k * polyvalfromroots(h, z) / polyvalfromroots(h, p)

    return w, h


def output2riemann_sphere(o: torch.Tensor, sections: int):
    return rearrange(
        o, "... batch (sections h) -> ... batch sections h", sections=sections
    )


def polyvalfromroots(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    r = r.reshape(r.shape + (1,) * x.ndim)

    return reduce(x - r, "... r x -> ... x", "prod")


def riemann_sphere2dft(r: torch.Tensor, N: int) -> torch.Tensor:
    zp = torch.exp(1j * r[..., ::2]) * (1 / torch.tan(0.5 * r[..., 1::2]))

    z = zp[..., :2]
    p = zp[..., 2:]

    _, h = freqz_zpk(z, p, 1, N)

    h = reduce(h, "... sections h -> ... h", "prod")

    return h
