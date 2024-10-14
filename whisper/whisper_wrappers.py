# SPDX-License-Identifier: GPL-3.0-or-later
#
# whisper_wrappers.py -- whisper wrappers
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>
# Copyright (C) 2024  Isaiah Rivera <isaiahcooperdump@gmail.com>

import os

import torch
import whisper

from einops import rearrange
from jiwer import (
    cer,
    wer,
)
from pytorch_lightning import LightningModule
from torch import (
    nn,
    optim,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchaudio.datasets import LIBRISPEECH
from whisper import (
    ModelDimensions,
    _ALIGNMENT_HEADS,
    _MODELS,
    _download,
)
from whisper.audio import (
    pad_or_trim,
    log_mel_spectrogram,
)
from whisper.decoding import DecodingOptions
from whisper.normalizers import EnglishTextNormalizer
from whisper.tokenizer import get_tokenizer


IGNORE_INDEX = -100
LANGUAGE = "en"
MODEL = "tiny.en"


class Collate:
    def __init__(self, normalizer, tokenizer):
        self.normalizer = normalizer
        self.tokenizer = tokenizer

    def __call__(self, batch):
        mel = []
        tokens = []

        for audio, transcript in batch:
            audio = pad_or_trim(audio)

            mel += [log_mel_spectrogram(audio)]

            tokens += [
                [*self.tokenizer.sot_sequence_including_notimestamps]
                + self.tokenizer.encode(self.normalizer(transcript))
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

        audio = audio.squeeze()

        return audio, transcript


class Whisper(LightningModule):
    def __init__(self, model, options, normalizer, tokenizer):
        super().__init__()

        self.model = model
        self.options = options
        self.normalizer = normalizer
        self.tokenizer = tokenizer

        self.loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.cer = cer
        self.wer = wer

        for p in self.model.parameters():
            p.requires_grad = False

        for p in self.model.decoder.parameters():
            p.requires_grad = True

    def _eval_step(self, batch, batch_idx, log=''):
        mel, tokens, labels = batch

        prediction = self.model.forward(mel, tokens)

        reference = self.decode(labels)
        hypothesis = self.decode(torch.argmax(prediction, axis=-1))

        cer = self.cer(reference, hypothesis)
        wer = self.wer(reference, hypothesis)

        prediction, labels = self._rearrange(prediction, labels)

        loss = self.loss(prediction, labels)

        if log:
            self.log(f"{log}/loss", loss, prog_bar=True)
            self.log(f"{log}/cer", cer, prog_bar=True)
            self.log(f"{log}/wer", wer, prog_bar=True)

    def _rearrange(self, prediction, labels):
        prediction = rearrange(prediction, "b t f -> (b t) f")
        labels = rearrange(labels, "b t -> (b t)")

        return prediction, labels

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=1e-6,
        )

        return optimizer

    @property
    def collate_fn(self):
        collate = Collate(self.normalizer, self.tokenizer)

        return collate

    def decode(self, tokens):
        tokens[tokens == IGNORE_INDEX] = self.tokenizer.eot

        transcripts = [self.tokenizer.decode(t).strip() for t in tokens]
        transcripts = [self.normalizer(t) for t in transcripts]

        return transcripts

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, 'test')

    def training_step(self, batch, batch_idx):
        mel, tokens, labels = batch

        prediction = self.model.forward(mel, tokens)

        prediction, labels = self._rearrange(prediction, labels)

        loss = self.loss(prediction, labels)

        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, 'val')


def load_base_model(name: str) -> Whisper:
    default = os.path.join(os.path.expanduser("~"), ".cache")
    download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")

    checkpoint_file = _download(_MODELS[name], download_root, False)
    alignment_heads = _ALIGNMENT_HEADS[name]

    with open(checkpoint_file, "rb") as f:
        checkpoint = torch.load(f, weights_only=True)

    del checkpoint_file

    dims = ModelDimensions(**checkpoint["dims"])

    model = whisper.Whisper(dims)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.set_alignment_heads(alignment_heads)

    return model


def load_model(name=MODEL, language=LANGUAGE):
    options = DecodingOptions(language=LANGUAGE, without_timestamps=True)

    normalizer = EnglishTextNormalizer()

    tokenizer = get_tokenizer(
        multilingual=False,
        language=LANGUAGE,
        task=options.task,
    )

    model = Whisper(
        load_base_model(MODEL),
        options,
        normalizer,
        tokenizer,
    )

    return model
