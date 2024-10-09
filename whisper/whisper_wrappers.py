# SPDX-License-Identifier: GPL-3.0-or-later
#
# whisper_wrappers.py -- whisper wrappers
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>
# Copyright (C) 2024  Isaiah Rivera <isaiahcooperdump@gmail.com>

import os

import torch

from torch.utils.data import Dataset
from torchaudio.datasets import LIBRISPEECH
from whisper import (
    ModelDimensions,
    Whisper,
    _ALIGNMENT_HEADS,
    _MODELS,
    _download,
)
from whisper.audio import pad_or_trim


class LibriSpeech(Dataset):
    def __init__(self, dataset):
        self.dataset = LIBRISPEECH(
            root=os.environ.get("DATASETS_PATH", "."),
            url=dataset,
            download=True,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, _, transcript, _, _, _ = self.dataset[item]

        audio = pad_or_trim(audio.squeeze())

        return audio, transcript


def load_base_model(name: str) -> Whisper:
    default = os.path.join(os.path.expanduser("~"), ".cache")
    download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")

    checkpoint_file = _download(_MODELS[name], download_root, False)
    alignment_heads = _ALIGNMENT_HEADS[name]

    with open(checkpoint_file, "rb") as f:
        checkpoint = torch.load(f, weights_only=True)

    del checkpoint_file

    dims = ModelDimensions(**checkpoint["dims"])

    model = Whisper(dims)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.set_alignment_heads(alignment_heads)

    return model
