#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
from pathlib import Path
from typing import Any, Literal

from deepspeed.profiling.flops_profiler import get_model_profile
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from torchoutil import move_to_rec

from dcase24t6.models.aac import TestBatch
from dcase24t6.utils.saving import save_to_yaml

pylog = logging.getLogger(__name__)


class OpCounter(Callback):
    def __init__(
        self,
        save_dir: str | Path,
        cplxity_fname: str = "model_complexity.yaml",
        backend: Literal["deepspeed"] = "deepspeed",
        verbose: int = 1,
    ) -> None:
        save_dir = Path(save_dir).resolve()
        super().__init__()
        self.save_dir = save_dir
        self.cplxity_fname = cplxity_fname
        self.backend = backend
        self.verbose = verbose

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        datamodule = trainer.datamodule  # type: ignore
        if "batch_size" not in datamodule.hparams:
            pylog.warning(
                "Cannot compute FLOPs or MACs since datamodule does not have batch_size hyperparameter."
            )
            return None

        source_batch_size = datamodule.hparams["batch_size"]
        target_batch_size = 1
        datamodule.hparams["batch_size"] = target_batch_size
        loaders = datamodule.test_dataloader()
        datamodule.hparams["batch_size"] = source_batch_size

        dataloader_idx = 0
        if isinstance(loaders, list):
            loader = loaders[dataloader_idx]
        else:
            loader = loaders
        del loaders
        batch: TestBatch = next(iter(loader))
        batch["captions"] = batch["mult_captions"][:, 0]  # type: ignore

        METHODS = ("forcing", "generate")
        metrics = {}
        for method in METHODS:
            example = {
                "batch": batch,
                "method": method,
            }

            start = time.perf_counter()
            match self.backend:
                case "deepspeed":
                    flops, macs, params = measure_complexity_with_deepspeed(
                        model=pl_module,
                        example=example,
                        verbose=self.verbose,
                    )
                case backend:
                    raise ValueError(f"Invalid argument {backend=}.")
            end = time.perf_counter()

            example_metrics = {
                "params": params,
                "flops": flops,
                "macs": macs,
                "duration": end - start,
            }
            metrics |= {f"{method}_{k}": v for k, v in example_metrics.items()}

        for pl_logger in pl_module.loggers:
            pl_logger.log_metrics(metrics)  # type: ignore

        cplxity_info = {
            "metrics": metrics,
            "batch_size": target_batch_size,
            "dataloader": "test",
            "dataloader_idx": dataloader_idx,
            "fname": batch["fname"],
            "dataset": batch["dataset"],
            "subset": batch["subset"],
        }
        cplxity_fpath = self.save_dir.joinpath(self.cplxity_fname)
        save_to_yaml(cplxity_info, cplxity_fpath)


def measure_complexity_with_deepspeed(
    model: LightningModule,
    example: Any,
    verbose: int = 0,
) -> tuple[int, int, int]:
    example = move_to_rec(example, device=model.device)
    flops, macs, params = get_model_profile(
        model,
        kwargs=example,
        print_profile=verbose >= 2,
        detailed=True,
        as_string=False,
    )
    return flops, macs, params  # type: ignore
