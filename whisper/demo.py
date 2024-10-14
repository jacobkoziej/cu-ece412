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
import torch

from IPython.display import Audio
from librosa.display import (
    specshow,
    waveshow,
)
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
model = load_model()

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
