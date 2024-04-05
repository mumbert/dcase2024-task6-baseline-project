#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp
from pathlib import Path

import torch
from torch import nn

from dcase24t6.models.trans_decoder import TransDecoderModel
from dcase24t6.nn.ckpt import BASELINE_REGISTER
from dcase24t6.pre_processes.cnext import ResampleMeanCNext
from dcase24t6.tokenization.aac_tokenizer import AACTokenizer


def dcase2024_baseline(
    model_name_or_path: str | Path = "baseline_weights",
    tokenizer_name_or_path: str | Path = "baseline_tokenizer",
    offline: bool = False,
    device: str | torch.device | None = "cuda_if_available",
    verbose: int = 0,
) -> nn.Sequential:
    pre_process = ResampleMeanCNext("cnext_bl", offline=offline, device=device)

    if osp.isfile(model_name_or_path):
        model_path = Path(model_name_or_path)
    else:
        model_name = str(model_name_or_path)
        model_path = BASELINE_REGISTER.get_ckpt_path(model_name)
        if not offline and not model_path.exists():
            BASELINE_REGISTER.download_ckpt(model_name, verbose=verbose)

    if osp.isfile(tokenizer_name_or_path):
        tokenizer_path = Path(tokenizer_name_or_path)
    else:
        tokenizer_name = str(tokenizer_name_or_path)
        tokenizer_path = BASELINE_REGISTER.get_ckpt_path(tokenizer_name)
        if not offline and not tokenizer_path.exists():
            BASELINE_REGISTER.download_ckpt(tokenizer_name, verbose=verbose)

    tokenizer = AACTokenizer.from_file(tokenizer_path)
    model = TransDecoderModel.load_from_checkpoint(model_path, tokenizer=tokenizer)
    pipeline = nn.Sequential(pre_process, model)

    return pipeline
