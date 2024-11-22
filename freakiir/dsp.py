# SPDX-License-Identifier: GPL-3.0-or-later
#
# dsp.py -- DSP helpers
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import torch

from einops import reduce
from torch import pi


def freqz_zpk(
    z: torch.Tensor,
    p: torch.Tensor,
    k: torch.Tensor,
    *,
    N: int = 512,
    whole: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert z.dtype == p.dtype
    assert z.device == p.device
    assert z.shape == p.shape

    assert k.device == z.device

    w = torch.linspace(0, pi * (2 if whole else 1), N).to(z.device)
    h = torch.exp(1j * w)

    k = k.reshape(k.shape + (1,) * (z.ndim - 1))

    h = k * polyvalfromroots(h, z) / polyvalfromroots(h, p)

    return w, h


def polyvalfromroots(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    r = r.reshape(r.shape + (1,) * x.ndim)

    return reduce(x - r, "... r x -> ... x", "prod")


def unwrap(x: torch.Tensor) -> torch.Tensor:
    y: torch.Tensor = x % (2 * pi)

    return torch.where(y > pi, 2 * pi - y, y)
