#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
#
# train.py -- whisper trainer
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import torch

from einops import rearrange
from pytorch_lightning import LightningModule
from torch import (
    nn,
    optim,
)
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


class WhisperFineTuner(LightningModule):
    def __init__(self, model, options):
        super().__init__()

        self.model = model
        self.options = options

        self.loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def training_step(self, batch, batch_idx):
        mel, tokens, labels = batch

        prediction = self.model.forward(mel, tokens)

        prediction = rearrange(prediction, "b t f -> (b t) f")
        labels = rearrange(labels, "b t -> (b t)")

        loss = self.loss(prediction, labels)

        self.log("train/loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())

        return optimizer
