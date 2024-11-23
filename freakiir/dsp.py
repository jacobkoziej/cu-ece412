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


def unwrap(
    x: torch.Tensor,
    *,
    discount: float | None = None,
    period: float = 2 * pi,
    axis: int = -1,
) -> torch.Tensor:
    if discount is None:
        discount = period / 2

    high: float = period / 2
    low: float = -high

    correction_slice: list[slice, ...] = [slice(None, None)] * x.ndim

    correction_slice[axis] = slice(1, None)

    correction_slice: tuple[slice, ...] = tuple(correction_slice)

    dd: torch.Tensor = torch.diff(x, axis=axis)
    ddmod: torch.Tensor = torch.remainder(dd - low, period) + low

    ph_correct: torch.Tensor = ddmod - dd
    ph_correct: torch.Tensor = torch.where(dd.abs() < discount, 0, ph_correct)

    x[correction_slice] = x[correction_slice] + ph_correct.cumsum(axis)

    return x
