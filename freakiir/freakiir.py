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

from dsp import freqz_zpk


class FreakIir(LightningModule):
    def __init__(
        self,
        inputs: int = 512,
        layers: int = 4,
        hidden_dimension: int = 4 * 512,
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

        self.layers.append(nn.Linear(hidden_dimension, sections * 4))

        self.loss = nn.MSELoss()

        self._output2zp = self._output2min_phase

    def _dft2mag(self, h: torch.Tensor) -> torch.Tensor:
        return 20 * torch.log10(h.abs())

    def _step(self, batch, batch_idx, log):
        zp, h = batch

        input = self._dft2mag(h)

        prediction = self.forward(input)

        h_prediction = self._zp2dft(prediction)

        output = self._dft2mag(h_prediction)

        loss = self.loss(output, input)

        self.log(f"{log}/loss", loss, prog_bar=True)

        return loss

    def _output2min_phase(self, output: torch.Tensor) -> torch.Tensor:
        sections = self.hparams.sections

        zp: torch.Tensor = rearrange(
            output,
            "... (sections pairs zp complex) -> ... sections pairs zp complex",
            sections=sections,
            pairs=2,
            zp=1,
            complex=2,
        )

        zp = zp[..., 0] + 1j * zp[..., 1]
        zp = torch.cat([zp, zp.conj()], axis=-1)

        return zp

    def _zp2dft(self, zp: torch.Tensor) -> torch.Tensor:
        z = zp[..., 0, :]
        p = zp[..., 1, :]
        k = torch.tensor(1, dtype=zp.real.dtype).to(zp.device)
        N = self.hparams.inputs

        _, h = freqz_zpk(z, p, k, N=N, whole=False)
        h = reduce(h, "... sections h -> ... h", "prod")

        return h

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=1e-3,
        )

        return optimizer

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)

        zp = self._output2zp(x)

        return zp

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")
