# SPDX-License-Identifier: GPL-3.0-or-later
#
# freakiir.py -- freakIIR: IIRNet but with phase
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>
# Copyright (C) 2024  Eric Eng <eric.eng@cooper.edu>

import torch

from einops import rearrange


def dft2input(f: torch.Tensor):
    z = torch.log10(f)
    z = torch.stack([10 * z.real, z.imag], axis=-1)

    return rearrange(z, "... w z -> ... (w z)")
