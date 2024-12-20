# SPDX-License-Identifier: GPL-3.0-or-later
#
# dataset.py -- datasets
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import builtins

import torch
import torchaudio

from pathlib import Path
from typing import Final

from einops import (
    rearrange,
    reduce,
)
from torch.utils.data import Dataset

from dsp import freqz_zpk
from generate import Generator


class Irc1059Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        *,
        N: torch.Tensor | int = 512,
    ) -> None:
        self.files: list[Path] = tuple(data_root.glob("**/*.wav"))

        self.N: int = N

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, item: int | slice) -> torch.Tensor:
        squeeze: bool = isinstance(item, int)

        if squeeze:
            item = slice(item, item + 1)

        audio: torch.Tensor

        audio = torch.cat(
            [
                torchaudio.load(file)[0].unsqueeze(0)
                for file in self.files[item]
            ]
        )
        audio = reduce(audio, "... channel time -> ... time", "mean")

        if squeeze:
            audio = audio.squeeze()

        N: int = self.N

        h: torch.Tensor = torch.fft.rfft(audio, n=N * 2)

        return h[..., :N]


class RandomDataset(Dataset):
    def __init__(
        self,
        generator: Generator,
        sections: int,
        *,
        all_pass: bool = False,
        epoch_size: int = 1 << 14,
        N: torch.Tensor | int = 512,
    ) -> None:
        assert sections >= 2

        self.generator: Final[Generator] = generator
        self.sections: Final[int] = sections

        self.all_pass: Final[bool] = all_pass
        self.epoch_size: Final[int] = epoch_size
        self.N: Final[torch.Tensor | int] = N

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(
        self, item: int | slice
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        pairs: int = self.sections * batch_size

        zp: torch.Tensor

        if self.all_pass:
            r: torch.Tensor
            theta: torch.Tensor

            r, theta = self.generator(pairs // 2, polar=True)

            r = r.unsqueeze(-1)
            theta = theta.unsqueeze(-1)

            r = torch.cat([1 / r, r], axis=-1)
            theta = torch.cat([theta, theta], axis=-1)

            zp = r * torch.exp(1j * theta)
            zp = zp.flatten()

        else:
            zp = self.generator(pairs)

        zp = rearrange(
            zp,
            "(batch sections pairs zp) -> batch sections pairs zp",
            batch=batch_size,
            sections=self.sections,
            pairs=2,
            zp=1,
        )
        zp = torch.cat([zp, zp.conj()], axis=-1)
        zp = zp.squeeze()

        z: torch.Tensor = zp[..., 0, :]
        p: torch.Tensor = zp[..., 1, :]
        k: torch.Tensor = (
            torch.norm(p, dim=-1) / torch.norm(z, dim=-1)
            if self.all_pass
            else torch.tensor(1, dtype=z.real.dtype)
        )

        h: torch.Tensor
        _, h = freqz_zpk(z, p, k, N=self.N, whole=False)
        h = reduce(h, "... sections h -> ... h", "prod")

        return zp, h
