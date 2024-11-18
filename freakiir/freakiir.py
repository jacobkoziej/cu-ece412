# SPDX-License-Identifier: GPL-3.0-or-later
#
# freakiir.py -- freakIIR: IIRNet but with phase
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>
# Copyright (C) 2024  Eric Eng <eric.eng@cooper.edu>

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

from dsp import freqz_zpk


class FreakIir(LightningModule):
    def __init__(
        self,
        inputs: int = 512,
        layers: int = 2,
        hidden_dimension: int = 16 * 512,
        sections: int = 32,
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

    def _output2zp(self, output: torch.Tensor) -> torch.Tensor:
        zp: torch.Tensor = rearrange(
            output,
            "... (sections pairs zp complex) -> ... sections pairs zp complex",
            sections=self.hparams.sections,
            pairs=2,
            zp=2,
            complex=2,
        )

        zp = zp[..., 0] + 1j * zp[..., 1]

        return zp

    def _step(self, batch, batch_idx, log):
        zp, h = batch

        input = 20 * torch.log10(h.abs())

        prediction = self.forward(input)
        prediction = self._output2zp(prediction)

        h_prediction = self._zp2dft(prediction)

        output = 20 * torch.log10(h_prediction.abs())

        loss = self.loss(output, input)

        self.log(f"{log}/loss", loss, prog_bar=True)

        return loss

    def _zp2dft(self, zp: torch.Tensor) -> torch.Tensor:
        z = zp[..., 0, :]
        p = zp[..., 1, :]

        _, h = freqz_zpk(z, p, 1, N=self.hparams.inputs, whole=True)
        h = reduce(h, "... sections h -> ... h", "prod")

        return h

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
