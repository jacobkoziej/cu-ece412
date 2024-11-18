# SPDX-License-Identifier: GPL-3.0-or-later
#
# generate.py -- generate random filters
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>
# Copyright (C) 2024  Eric Eng <eric.eng@cooper.edu>

import torch

from collections.abc import Callable
from typing import (
    NewType,
    Optional,
)


Generator = NewType(
    "Generator",
    Callable[[int, float, Optional[torch.dtype]], torch.Tensor],
)


def uniform_half_disk(
    pairs: int,
    *,
    epsilon: float = 1e-6,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    from torch.distributions import Uniform

    if dtype is None:
        dtype = torch.get_default_dtype()

    samples: tuple[int] = (pairs * 2,)

    r_uniform: Uniform = Uniform(0 + epsilon, 1 - epsilon)
    theta_uniform: Uniform = Uniform(0, torch.pi - torch.finfo(dtype).eps)

    r: torch.Tensor = r_uniform.sample(samples)
    theta: torch.Tensor = theta_uniform.sample(samples)

    return r * torch.exp(1j * theta)
