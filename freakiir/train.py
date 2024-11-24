#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# train.py -- freakIIR trainer
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>


def main() -> None:
    import argparse
    import os

    import torch

    from pathlib import Path

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader

    from dataset import (
        Irc1059Dataset,
        RandomDataset,
    )
    from freakiir import FreakIir
    from generate import uniform_half_disk

    argparser = argparse.ArgumentParser(
        description="freakIRR trainer",
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
        "--ckpt",
        default=None,
        help="checkpoint",
        metavar="PATH",
        type=str,
    )
    argparser.add_argument(
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
    argparser.add_argument(
        "--order",
        default=8,
        help="filter order",
        metavar="N",
        type=int,
    )

    args = argparser.parse_args()

    assert not args.order % 2

    sections = args.order // 2

    model = (
        FreakIir.load_from_checkpoint(args.ckpt)
        if args.ckpt
        else FreakIir(sections=sections, all_pass=False)
    )

    random_dataset = RandomDataset(
        generator=uniform_half_disk,
        sections=sections,
        all_pass=False,
    )
    irc_1059_dataset = Irc1059Dataset(
        Path(f'{os.environ.get("DATASETS_PATH", ".")}/IRC_1059/COMPENSATED'),
    )

    num_workers = 4

    if torch.cuda.is_available():
        num_workers *= torch.cuda.device_count()

    random_loader = DataLoader(
        random_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )
    irc_1059_loader = DataLoader(
        irc_1059_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
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
        train_dataloaders=random_loader,
        val_dataloaders=irc_1059_loader,
        ckpt_path="last",
    )


if __name__ == "__main__":
    main()
