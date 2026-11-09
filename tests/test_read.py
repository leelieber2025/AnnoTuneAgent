# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

import gzip
from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData
from scipy import io, sparse

from annotuneagent.read import read_10x


def _write_mtx(folder: Path, genes: list[str], barcodes: list[str], counts) -> None:
    folder.mkdir(parents=True)
    io.mmwrite(folder / "matrix.mtx", sparse.csr_matrix(counts).T)
    (folder / "barcodes.tsv").write_text("".join(f"{name}\n" for name in barcodes))
    lines = [f"id{i}\t{name}\tGene Expression\n" for i, name in enumerate(genes)]
    (folder / "features.tsv").write_text("".join(lines))
    for name in ("matrix.mtx", "barcodes.tsv", "features.tsv"):
        path = folder / name
        with (
            path.open("rb") as source,
            gzip.open(path.with_suffix(path.suffix + ".gz"), "wb") as target,
        ):
            target.write(source.read())
        path.unlink()


def test_read_10x_concatenates_mtx_directories(tmp_path):
    _write_mtx(
        tmp_path / "A" / "filtered_feature_bc_matrix",
        ["GeneA", "GeneB"],
        ["c1", "c2"],
        np.array([[1, 0], [0, 3]]),
    )
    _write_mtx(
        tmp_path / "B" / "outs" / "filtered_feature_bc_matrix",
        ["GeneB", "GeneC"],
        ["c1"],
        np.array([[4, 5]]),
    )
    adata = read_10x(tmp_path, ["A", "B"])
    assert adata.n_obs == 3
    assert list(adata.var_names) == ["GeneA", "GeneB", "GeneC"]
    assert list(adata.obs["sample"]) == ["A", "A", "B"]
    assert list(adata.obs_names) == ["c1-A", "c2-A", "c1-B"]
    assert adata.obs["sample"].dtype.name == "category"


def test_read_10x_uses_h5_when_both_are_present(tmp_path, monkeypatch):
    sample = tmp_path / "A"
    _write_mtx(
        sample / "filtered_feature_bc_matrix",
        ["GeneA"],
        ["c1"],
        np.array([[1]]),
    )
    h5 = sample / "filtered_feature_bc_matrix.h5"
    h5.write_bytes(b"h5")
    seen = {}

    def fake_h5(path):
        seen["path"] = Path(path)
        adata = AnnData(np.array([[7]]))
        adata.obs_names = ["c9"]
        adata.var_names = ["FromH5"]
        return adata

    monkeypatch.setattr("annotuneagent.read.sc.read_10x_h5", fake_h5)
    adata = read_10x(tmp_path, ["A"])
    assert seen["path"] == h5
    assert list(adata.var_names) == ["FromH5"]
    assert list(adata.obs_names) == ["c9-A"]


def test_read_10x_makes_gene_and_cell_names_unique(tmp_path):
    _write_mtx(
        tmp_path / "A" / "filtered_feature_bc_matrix",
        ["GeneA", "GeneA"],
        ["c1", "c1"],
        np.array([[1, 2], [3, 4]]),
    )
    adata = read_10x(tmp_path, ["A"])
    assert list(adata.var_names) == ["GeneA", "GeneA-1"]
    assert len(set(adata.obs_names)) == adata.n_obs


def test_read_10x_reports_a_missing_matrix(tmp_path):
    (tmp_path / "A").mkdir()
    with pytest.raises(FileNotFoundError, match="filtered_feature_bc_matrix"):
        read_10x(tmp_path, ["A"])
