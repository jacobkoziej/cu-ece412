#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
#
# train.py -- whisper trainer
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>
# Copyright (C) 2024  Isaiah Rivera <isaiahcooperdump@gmail.com>

import argparse

import torch

from einops import rearrange
from jiwer import (
    cer,
    wer,
)
from pytorch_lightning import (
    LightningModule,
    Trainer,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import (
    nn,
    optim,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from whisper.audio import log_mel_spectrogram
from whisper.decoding import DecodingOptions
from whisper.normalizers import EnglishTextNormalizer
from whisper.tokenizer import get_tokenizer
from whisper_wrappers import (
    LibriSpeech,
    load_base_model,
)


IGNORE_INDEX = -100
LANGUAGE = "en"
MODEL = "tiny.en"


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
    def __init__(self, model, options, tokenizer):
        super().__init__()

        self.model = model
        self.options = options
        self.tokenizer = tokenizer

        for p in self.model.parameters():
            p.requires_grad = False

        for p in self.model.decoder.parameters():
            p.requires_grad = True

        self.loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.normalizer = EnglishTextNormalizer()
        self.cer = cer
        self.wer = wer

    def _decode(self, tokens):
        tokens[tokens == IGNORE_INDEX] = self.tokenizer.eot

        transcripts = [self.tokenizer.decode(t).strip() for t in tokens]
        transcripts = [self.normalizer(t) for t in transcripts]

        return transcripts

    def _rearrange(self, prediction, labels):
        prediction = rearrange(prediction, "b t f -> (b t) f")
        labels = rearrange(labels, "b t -> (b t)")

        return prediction, labels

    def training_step(self, batch, batch_idx):
        mel, tokens, labels = batch

        prediction = self.model.forward(mel, tokens)

        prediction, labels = self._rearrange(prediction, labels)

        loss = self.loss(prediction, labels)

        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        mel, tokens, labels = batch

        prediction = self.model.forward(mel, tokens)

        reference = self._decode(torch.argmax(prediction, axis=-1))
        hypothesis = self._decode(labels)

        cer = self.cer(reference, hypothesis)
        wer = self.wer(reference, hypothesis)

        prediction, labels = self._rearrange(prediction, labels)

        loss = self.loss(prediction, labels)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/cer", cer, prog_bar=True)
        self.log("val/wer", wer, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())

        return optimizer


def main() -> None:
    argparser = argparse.ArgumentParser(
        description="whisper fine-tuner",
    )

    argparser.add_argument(
        "-b",
        "--batch-size",
        default=16,
        help="batch size",
        metavar="N",
    )
    argparser.add_argument(
        "-c",
        "--checkpoint-path",
        default='checkpoints',
        help="checkpoint path",
        metavar="PATH",
    )
    argparser.add_argument(
        "-e",
        "--epochs",
        default=2,
        help="epochs",
        metavar="N",
    )

    args = argparser.parse_args()

    options = DecodingOptions(language=LANGUAGE, without_timestamps=True)

    tokenizer = get_tokenizer(
        multilingual=False,
        language=LANGUAGE,
        task=options.task,
    )

    train = LibriSpeech("train-clean-100")

    collate = Collate(tokenizer)

    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=torch.cuda.device_count() * 4,
    )

    val = LibriSpeech("dev-clean")

    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=torch.cuda.device_count() * 4,
    )

    model = WhisperFineTuner(
        load_base_model(MODEL),
        options,
        tokenizer=tokenizer,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        save_last='link',
        save_top_k=-1,
        every_n_epochs=1,
    )

    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=args.epochs,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path='last',
    )


if __name__ == "__main__":
    main()
