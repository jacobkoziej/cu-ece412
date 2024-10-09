#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
#
# train.py -- whisper trainer
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import torch

from torch.nn.utils.rnn import pad_sequence
from whisper.audio import log_mel_spectrogram


IGNORE_INDEX = -100


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        mel = []
        tokens = []

        for audio, transcript in batch:
            mel += [log_mel_spectrogram(audio)]
            tokens += [
                [*self.tokenizer.sot_sequence_including_notimestamps]
                + self.tokenizer.encode(transcript)
            ]

        mel = torch.stack(mel)

        labels = [torch.tensor(t[1:] + [self.tokenizer.eot]) for t in tokens]
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        tokens = pad_sequence(
            [torch.tensor(t) for t in tokens],
            batch_first=True,
            padding_value=self.tokenizer.eot,
        )

        return mel, tokens, labels
