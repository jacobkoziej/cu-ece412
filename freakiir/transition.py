# SPDX-License-Identifier: GPL-3.0-or-later
#
# transition.py -- transition maps
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import torch

from torch import pi


def horn_torus2zplane(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    phi = torch.where(phi >= 0, phi, 2 * pi + phi)

    r: torch.Tensor = torch.exp(-1 / torch.tan(phi / 2))
    z: torch.Tensor = r * torch.exp(1j * theta)

    return z


def zplane2horn_torus(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    theta: torch.Tensor = z.angle()

    phi: torch.Tensor
    phi = 2 * (pi / 2 + torch.arctan(torch.log(z.abs())))
    phi = torch.where(phi <= pi, phi, -(2 * pi - phi))

    return theta, phi
