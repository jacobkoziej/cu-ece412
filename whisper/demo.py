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
# Copyright (C) 2024  Isaiah Rivera <isaiahcooperdump@gmail.com>

# %% [markdown]
# # Setup

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from IPython.display import Audio
from librosa.display import (
    specshow,
    waveshow,
)
from scipy.signal import lfilter
from torch_audiomentations import (
    AddColoredNoise,
    Compose,
)
from whisper.audio import (
    log_mel_spectrogram,
    pad_or_trim,
)

from whisper_wrappers import (
    LibriSpeech,
    load_model,
)

# %%
model = load_model("checkpoints/ideal.ckpt")
model.eval()

# %%
test = LibriSpeech("test-clean")

# %% [markdown]
# ## Audio Pipeline

# %%
sample_rate = 16_000
audio, text = test[0]

# %% tags=["active-ipynb"]
Audio(audio, rate=sample_rate)

# %% tags=["active-ipynb"]
text

# %% [markdown]
# ### Padding
#
# Whisper expects up to 30s audio segments so here we'll zero-pad or
# trim.

# %%
# Comment the following line for better plots.
audio = pad_or_trim(audio)

# %% tags=["active-ipynb"]
_ = waveshow(audio.numpy(), sr=sample_rate, max_points=len(audio))

# %% [markdown]
# ### Log Mel Spectrogram
#
# Whisper expects indirectly analyses audio using a 2D convolution of a
# logarithmic Mel spectrogram.

# %%
mel = log_mel_spectrogram(audio)

# %% tags=["active-ipynb"]
_ = specshow(mel.numpy(), sr=sample_rate)

# %% [markdown]
# ### Normalization
#
# To make it easier to fine-tune Whisper, we apply a text normalizer to
# better match the already trained ASR content. This turns out to be an
# important step to speed up training.

# %%
text = model.normalizer(text)

# %% tags=["active-ipynb"]
text

# %% [markdown]
# ### Tokenization
#
# We now generate two token streams: labels which we can use for
# evaluating loss and tokens that we'll feed into the model for
# training.

# %%
tokens = [
    *model.tokenizer.sot_sequence_including_notimestamps
] + model.tokenizer.encode(text)

# %% tags=["active-ipynb"]
tokens

# %% [markdown]
# We remove the SOT token from our labels and pad the end with our
# `IGNORE_INDEX` to prevent inaccurate loss calculations. While for our
# training tokens, we keep SOT and append EOT.

# %%
IGNORE_INDEX = -100

labels = torch.tensor(tokens[1:] + [IGNORE_INDEX])
tokens = torch.tensor(tokens + [model.tokenizer.eot])

# %% [markdown]
# # Evaluation

# %%
# add a dummy batch dimension to make Whisper happy
mel = mel.unsqueeze(0)
tokens = tokens.unsqueeze(0)
labels = labels.unsqueeze(0)

prediction = model.forward(mel, tokens)

reference = model.decode(labels)
hypothesis = model.decode(torch.argmax(prediction, axis=-1))

cer = model.cer(reference, hypothesis)
wer = model.wer(reference, hypothesis)

# %% tags=["active-ipynb"]
reference

# %% tags=["active-ipynb"]
hypothesis

# %% tags=["active-ipynb"]
cer

# %% tags=["active-ipynb"]
wer

# %% [markdown]
# ## Pushing Limits
#
# Let's ruin the SNR to make things more interesting!

# %%
augment = Compose(
    [
        AddColoredNoise(
            min_snr_in_db=-10,
            max_snr_in_db=-10,
            min_f_decay=0,
            max_f_decay=0,
            output_type="tensor",
        ),
    ],
    output_type="tensor",
)

# %%
audio = audio.unsqueeze(0)
audio = audio.unsqueeze(0)
audio = augment(audio, sample_rate)
audio = audio.squeeze()

# %% tags=["active-ipynb"]
Audio(audio, rate=sample_rate)

# %%
mel = log_mel_spectrogram(audio).unsqueeze(0)

prediction = model.forward(mel, tokens)

reference = model.decode(labels)
hypothesis = model.decode(torch.argmax(prediction, axis=-1))

cer = model.cer(reference, hypothesis)
wer = model.wer(reference, hypothesis)

# %% tags=["active-ipynb"]
reference

# %% tags=["active-ipynb"]
hypothesis

# %% tags=["active-ipynb"]
cer

# %% tags=["active-ipynb"]
wer

# %% [markdown]
# Needless to say, impressive!

# %% [markdown]
# # Results


# %%
def ax_stem(ax, log, type, name, linefmt):

    ax.stem(log["step"], log["value"], label=f"{type}/{name}", linefmt=linefmt)


# %%
def lpf_plot(ax, log, type, name, color):
    alpha = 0.6

    value = lfilter([1 - alpha], [1, -alpha], log["value"])

    ax.plot(log["step"], value, label=f"{type}/{name}", color=color)


# %%
stem_logs = [
    {
        "type": "val",
        "name": "cer",
        "linefmt": "b--",
    },
    {
        "type": "val",
        "name": "wer",
        "linefmt": "r--",
    },
]

# %%
ax = plt.subplot()

for stem_log in stem_logs:
    log = pd.read_csv(f"logs/{stem_log['type']}/{stem_log['name']}.csv")
    ax_stem(ax, log, **stem_log)

epochs = len(log["step"])
xticks = np.linspace(log["step"].iloc[0], log["step"].iloc[-1], epochs)
xlabels = [str(i) for i in range(epochs)]

_ = ax.set_title("Error Rates")
_ = ax.set_xticks(xticks, xlabels)
_ = ax.set_xlabel("Epoch")
_ = ax.set_ylabel("Error")
_ = ax.legend()

# %%
ax = plt.subplot()

log = pd.read_csv("logs/val/loss.csv")
ax_stem(ax, log, "val", "loss", "r--")

log = pd.read_csv("logs/train/loss.csv")
lpf_plot(ax, log, "train", "loss", "blue")

_ = ax.set_title("Loss Rates")
_ = ax.set_xlabel("Step")
_ = ax.set_ylabel("Loss")
_ = ax.legend()

# %% [markdown]
# ```
# baseline/test/cer  0.1266426295042038
# baseline/test/loss 1.1183173656463623
# baseline/test/wer  0.1837381273508072
#
# ideal/test/cer     0.03146420791745186
# ideal/test/loss    0.8304051160812378
# ideal/test/wer     0.05836161971092224
# ```
