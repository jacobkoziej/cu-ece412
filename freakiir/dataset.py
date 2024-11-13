# SPDX-License-Identifier: GPL-3.0-or-later
#
# dataset.py -- datasets
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import builtins

import torch

from typing import Final

from einops import rearrange
from torch.utils.data import Dataset

from generate import Generator


class RandomDataset(Dataset):
    def __init__(
        self,
        generator: Generator,
        sections: int,
        epoch_size: int = 1 << 14,
        *,
        w_or_N: torch.Tensor | int = 512,
    ) -> None:
        self.epoch_size: Final[int] = epoch_size
        self.sections: Final[int] = sections
        self.generator: Final[Generator] = generator
        self.w_or_N: Final[torch.Tensor | int] = w_or_N

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, item: int | slice) -> torch.Tensor:
        batch_size: int

        match type(item):
            case builtins.int:
                batch_size = 1

            case builtins.slice:
                start: int = item.start if item.start else 0
                stop: int = item.stop if item.stop else self.epoch_size
                step: int = item.step if item.step else 1

                batch_size = abs((stop - start) // step)

            case _:
                raise TypeError("index must be int or slice")

        pairs: int = 2 * self.sections * batch_size

        batch: torch.Tensor = rearrange(
            self.generator(pairs),
            "(batch sections pairs zp) -> batch sections pairs zp",
            batch=batch_size,
            sections=self.sections,
            pairs=2,
            zp=2,
        )

        return batch.squeeze()
