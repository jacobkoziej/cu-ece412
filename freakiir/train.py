#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# train.py -- freakIIR trainer
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>


def main() -> None:
    import argparse
    import os

    import pandas as pd
    import torch

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader

    from freakiir import (
        FreakIir,
        FreakIirDataset,
    )

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
        default=4,
        help="filter order",
        metavar="N",
        type=int,
    )
    argparser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="test",
    )
    argparser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="validate",
    )

    args = argparser.parse_args()

    assert not args.order % 2

    sections = args.order // 2

    model = FreakIir(sections=sections)

    dataset_root = os.path.join(
        os.environ.get("DATASETS_PATH", "."), f"freakIIR/{args.order}"
    )

    datasets = ["train", "test", "val"]
    datasets = {
        dataset: FreakIirDataset(
            torch.tensor(
                pd.read_csv(os.path.join(dataset_root, f"{dataset}.csv")).values,
                dtype=torch.float32,
            ),
            sections,
        )
        for dataset in datasets
    }

    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=torch.cuda.device_count() * 4,
        )
        for (name, dataset) in datasets.items()
    }

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

    if args.test:
        trainer.test(model, loaders["test"])
        exit()

    if args.validate:
        trainer.validate(model, loaders["test"])
        exit()

    trainer.fit(
        model,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["val"],
        ckpt_path="last",
    )


if __name__ == "__main__":
    main()
