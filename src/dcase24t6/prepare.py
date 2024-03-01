#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["TRANSFORMERS_OFFLINE"] = "FALSE"
os.environ["HF_HUB_OFFLINE"] = "FALSE"

import logging
import os
import os.path as osp
from pathlib import Path
from typing import Callable, Iterable, Literal

import hydra
import nltk
from aac_datasets.datasets.clotho import Clotho
from aac_datasets.datasets.functional.clotho import download_clotho_datasets
from aac_metrics.download import download_metrics
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataset import Subset
from torchoutil.utils.hdf import pack_to_hdf

pylog = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=osp.join("..", "conf"),
    config_name="prepare",
)
def prepare(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    if cfg.verbose >= 1:
        pylog.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    pre_process = instantiate(cfg.pre_process)

    return prepare_data_metrics_models(
        dataroot=cfg.path.data_root,
        subsets=cfg.subsets,
        download_clotho=cfg.download_clotho,
        force=cfg.force,
        hdf_pattern=cfg.hdf_pattern,
        pre_process=pre_process,
        overwrite=cfg.overwrite,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        size_limit=cfg.size_limit,
        verbose=cfg.verbose,
    )


def prepare_data_metrics_models(
    dataroot: str | Path = "data",
    subsets: Iterable[str] = (),
    download_clotho: bool = True,
    force: bool = False,
    hdf_pattern: str = "{dataset}_{subset}.hdf",
    pre_process: Callable | None = None,
    overwrite: bool = False,
    batch_size: int = 32,
    num_workers: int | Literal["auto"] = "auto",
    size_limit: int | None = None,
    verbose: int = 0,
) -> None:
    dataroot = Path(dataroot).resolve()
    subsets = list(subsets)

    nltk.download("stopwords")
    download_metrics(verbose=verbose)

    os.makedirs(dataroot, exist_ok=True)

    if download_clotho:
        download_clotho_datasets(
            root=dataroot,
            subsets=subsets,
            force=force,
            verbose=verbose,
            clean_archives=False,
            verify_files=True,
        )

    hdf_root = dataroot.joinpath("HDF")
    os.makedirs(hdf_root, exist_ok=True)

    for subset in subsets:
        dataset = Clotho(
            root=dataroot,
            subset=subset,
            download=False,
            verbose=verbose,
        )

        if size_limit is not None and len(dataset) > size_limit:
            dataset = Subset(dataset, list(range(size_limit)))

        # example: clotho_dev.hdf
        hdf_fname = hdf_pattern.format(
            dataset="clotho",
            subset=subset,
        )
        hdf_fpath = hdf_root.joinpath(hdf_fname)

        if hdf_fpath.exists() and not overwrite:
            continue

        pack_to_hdf(
            dataset=dataset,  # type: ignore
            hdf_fpath=hdf_fpath,
            pre_transform=pre_process,
            overwrite=overwrite,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
        )


if __name__ == "__main__":
    prepare()
