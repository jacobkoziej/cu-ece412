#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
#
# train.py -- whisper trainer
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>
# Copyright (C) 2024  Isaiah Rivera <isaiahcooperdump@gmail.com>

import argparse

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from whisper_wrappers import (
    LibriSpeech,
    load_model,
)


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
        type=int,
    )
    argparser.add_argument(
        "-c",
        "--checkpoint-path",
        default="checkpoints",
        help="checkpoint path",
        metavar="PATH",
        type=str,
    )
    argparser.add_argument(
        "-e",
        "--epochs",
        default=-1,
        help="epochs",
        metavar="N",
        type=int,
    )

    args = argparser.parse_args()

    model = load_model()

    train = LibriSpeech("train-clean-100")
    val = LibriSpeech("dev-clean")

    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        collate_fn=model.collate_fn,
        num_workers=torch.cuda.device_count() * 4,
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        collate_fn=model.collate_fn,
        num_workers=torch.cuda.device_count() * 4,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        save_last="link",
        save_top_k=-1,
        every_n_epochs=1,
    )

    trainer = Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps=args.batch_size,
        max_epochs=args.epochs,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path="last",
    )


if __name__ == "__main__":
    main()
