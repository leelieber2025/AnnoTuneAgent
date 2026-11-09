# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Read 10x count matrices from a sample folder."""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import scanpy as sc
from anndata import AnnData

_H5_NAME = "filtered_feature_bc_matrix.h5"
_MTX_NAME = "filtered_feature_bc_matrix"


def read_10x(base: str | Path, samples: list[str], *, sample_key: str = "sample") -> AnnData:
    """Read one filtered 10x matrix per sample and concatenate them.

    ``base`` is the folder that contains one folder per sample name. Each
    sample folder is searched for ``filtered_feature_bc_matrix.h5`` or a
    ``filtered_feature_bc_matrix`` directory. An h5 file is used when both
    are present. Barcodes get the sample name as a suffix. Gene names and
    cell names are made unique.
    """
    root = Path(base)
    names = [str(name) for name in samples]
    if not names:
        raise ValueError("samples is empty")
    if len(set(names)) != len(names):
        raise ValueError("sample names must be unique")
    pieces: dict[str, AnnData] = {}
    for name in names:
        folder = root / name
        if not folder.is_dir():
            raise FileNotFoundError(f"sample folder is missing: {folder}")
        path = _find_matrix(folder)
        piece = _unique(_read_matrix(path))
        print(f"{name}: {piece.shape}  {path}", flush=True)
        pieces[name] = piece
    merged = ad.concat(pieces, label=sample_key, index_unique="-", join="outer", merge="same")
    merged.obs[sample_key] = merged.obs[sample_key].astype("category")
    return _unique(merged)


def _find_matrix(folder: Path) -> Path:
    found: list[Path] = []
    for current, dirnames, filenames in os.walk(folder):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        here = Path(current)
        if _H5_NAME in filenames:
            found.append(here / _H5_NAME)
        mtx = here / _MTX_NAME
        if _MTX_NAME in dirnames and _is_mtx(mtx):
            found.append(mtx)
    if not found:
        raise FileNotFoundError(f"{folder} has no {_H5_NAME} or {_MTX_NAME}")
    return min(found, key=lambda path: (len(path.parts), path.suffix != ".h5", str(path)))


def _is_mtx(folder: Path) -> bool:
    return any((folder / name).is_file() for name in ("matrix.mtx", "matrix.mtx.gz"))


def _unique(adata: AnnData) -> AnnData:
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata


def _read_matrix(path: Path) -> AnnData:
    if path.is_dir():
        return sc.read_10x_mtx(path, var_names="gene_symbols", make_unique=True)
    return sc.read_10x_h5(path)
