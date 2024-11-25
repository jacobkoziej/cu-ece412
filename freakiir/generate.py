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
    Callable[
        [int, float, Optional[torch.dtype], ...],
        torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ],
)


def uniform_half_disk(
    pairs: int,
    *,
    epsilon: float = 1e-6,
    polar: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

    return uniform_half_ring(
        pairs=pairs,
        lower_limit=0.0,
        upper_limit=1.0,
        epsilon=epsilon,
        polar=polar,
        dtype=dtype,
    )


def uniform_half_ring(
    pairs: int,
    *,
    lower_limit: float = 0.5,
    upper_limit: float = 1.0,
    epsilon: float = 1e-6,
    polar: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    assert lower_limit <= upper_limit

    assert lower_limit >= 0.0
    assert upper_limit <= 1.0

    from torch.distributions import Uniform

    if dtype is None:
        dtype = torch.get_default_dtype()

    samples: tuple[int] = (pairs * 2,)

    r_uniform: Uniform = Uniform(lower_limit + epsilon, upper_limit - epsilon)
    theta_uniform: Uniform = Uniform(0, torch.pi - torch.finfo(dtype).eps)

    r: torch.Tensor = r_uniform.sample(samples)
    theta: torch.Tensor = theta_uniform.sample(samples)

    if polar:
        return (r, theta)

    return r * torch.exp(1j * theta)
