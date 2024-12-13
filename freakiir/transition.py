# SPDX-License-Identifier: GPL-3.0-or-later
#
# transition.py -- transition maps
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import torch

from torch import pi


def zplane2horn_torus(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    theta: torch.Tensor = z.angle()

    phi: torch.Tensor
    phi = 2 * (pi / 2 + torch.arctan(torch.log(z.abs())))
    phi = torch.where(phi <= pi, phi, -(2 * pi - phi))

    return theta, phi
