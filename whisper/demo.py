# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# SPDX-License-Identifier: GPL-3.0-or-later
#
# demo.py -- OpenAI's Whisper on LibriSpeech ASR
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

# %%
import os

from torchaudio.datasets import LIBRISPEECH

)
# %%
train = LibriSpeech("train-clean-100")
validate = LibriSpeech("dev-clean")
test = LibriSpeech("test-clean")
