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
from scipy.signal import freqz_zpk
from torch.utils.data import Dataset


class FreakIirDataset(Dataset):
    def __init__(self, dataset: torch.Tensor, order: int, *, N: int = 512):
        assert order >= 2
        assert not order % 2

        self.riemann_sphere = rearrange(
            dataset, "(batch order) h -> batch order h", order=order
        )

        zp = torch.exp(1j * dataset[..., ::2]) * (
            1 / torch.tan(0.5 * dataset[..., 1::2])
        )
        zp = zp.numpy()

        z = zp[..., :2]
        p = zp[..., 2:]

        h = torch.tensor(
            np.array([freqz_zpk(z, p, 1, worN=N, whole=True)[-1] for z, p in zip(z, p)])
        )
        h = rearrange(h, "(batch order) h -> batch order h", order=order)
        h = reduce(h, "batch order h -> batch h", "prod")

        self.h = h

        self.input = dft2input(h)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        riemann_sphere = self.riemann_sphere[item]
        h = self.h[item]
        input = self.input[item]

        return riemann_sphere, h, input


def dft2input(f: torch.Tensor):
    z = torch.log10(f)
    z = torch.stack([10 * z.real, z.imag], axis=-1)

    return rearrange(z, "... w z -> ... (w z)")
