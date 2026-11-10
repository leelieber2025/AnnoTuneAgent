# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

import gzip
from pathlib import Path

import numpy as np
import pytest
from scipy import io, sparse

from annotuneagent.geo import geo_info, read_geo


def _write_mtx(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    io.mmwrite(folder / "matrix.mtx", sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 3.0]])).T)
    (folder / "barcodes.tsv").write_text("c1\nc2\n")
    (folder / "features.tsv").write_text(
        "id0\tGeneA\tGene Expression\nid1\tGeneB\tGene Expression\n"
    )
    for name in ("matrix.mtx", "barcodes.tsv", "features.tsv"):
        path = folder / name
        with (
            path.open("rb") as source,
            gzip.open(path.with_suffix(path.suffix + ".gz"), "wb") as out,
        ):
            out.write(source.read())
        path.unlink()


def _soft(*samples: dict) -> str:
    lines = [
        "^SERIES = GSE1",
        "!Series_title = Example series",
        "!Series_summary = Three replicates.",
        "!Series_sample_organism = Mus musculus",
    ]
    for sample in samples:
        gsm = sample["gsm"]
        lines.append(f"^SAMPLE = {gsm}")
        lines.append(f"!Sample_title = {sample['title']}")
        lines.append(f"!Sample_organism_ch1 = {sample.get('organism', 'Mus musculus')}")
        for key, value in sample.get("chars", {}).items():
            lines.append(f"!Sample_characteristics_ch1 = {key}: {value}")
        for name in sample.get("files", ("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz")):
            lines.append(f"!Sample_supplementary_file = https://example.test/{gsm}_{name}")
    return "\n".join(lines)


def _patch(monkeypatch, text: str) -> None:
    monkeypatch.setattr("annotuneagent.geo._fetch_soft", lambda accession, root: text)


def _cache_matrices(cache: Path, *gsms: str) -> None:
    for gsm in gsms:
        _write_mtx(cache / "GSE1" / gsm)


def test_geo_info_lists_samples_and_does_not_download_counts(tmp_path, monkeypatch):
    text = _soft(
        {"gsm": "GSM1", "title": "B1", "chars": {"treatment": "C&AIA", "strain": "C57BL/6"}},
        {"gsm": "GSM2", "title": "B2", "chars": {"treatment": "C&AIA", "strain": "C57BL/6"}},
        {"gsm": "GSM3", "title": "B3", "files": ("readme.txt",), "chars": {"treatment": "C&AIA"}},
    )
    _patch(monkeypatch, text)
    seen: list[str] = []
    monkeypatch.setattr(
        "annotuneagent.geo._materialize",
        lambda sample, root: seen.append(sample.title),
    )
    info = geo_info("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE1", cache=tmp_path)
    assert seen == []
    assert info.accession == "GSE1"
    assert info.organism == "Mus musculus"
    assert [(item.title, item.accession, item.counts) for item in info.samples] == [
        ("B1", "GSM1", "mtx"),
        ("B2", "GSM2", "mtx"),
        ("B3", "GSM3", ""),
    ]
    assert info.samples[0].characteristics["strain"] == "C57BL/6"
    printed = str(info)
    assert "B1  GSM1  10x mtx" in printed
    assert "treatment: C&AIA" in printed
    assert "B3  GSM3  no count matrix" in printed


def test_read_geo_downloads_only_named_samples(tmp_path, monkeypatch):
    text = _soft(
        {"gsm": "GSM1", "title": "B1", "chars": {"treatment": "C&AIA"}},
        {"gsm": "GSM2", "title": "B2", "chars": {"treatment": "C&AIA"}},
        {"gsm": "GSM3", "title": "B3", "chars": {"treatment": "C&AIA"}},
    )
    _patch(monkeypatch, text)
    _cache_matrices(tmp_path, "GSM1", "GSM2", "GSM3")
    seen: list[str] = []
    real = __import__("annotuneagent.geo", fromlist=["_materialize"])._materialize

    def spy(sample, root):
        seen.append(sample.title)
        return real(sample, root)

    monkeypatch.setattr("annotuneagent.geo._materialize", spy)
    info = geo_info("GSE1", cache=tmp_path)
    adata = read_geo(
        info,
        samples="B3, GSM1",
        groups="B1：contro，B3: Day7",
        cache=tmp_path,
    )
    assert seen == ["B3", "B1"]
    assert list(adata.obs["sample"].astype(str)) == ["B3", "B3", "B1", "B1"]
    assert list(adata.obs["group"].astype(str)) == ["Day7", "Day7", "contro", "contro"]
    assert list(adata.obs["geo_accession"].astype(str).unique()) == ["GSM3", "GSM1"]
    assert adata.uns["geo"]["groups"] == {"B3": "Day7", "B1": "contro"}
    assert adata.uns["geo"]["organism"] == "Mus musculus"


def test_unknown_sample_and_group_names_raise(tmp_path, monkeypatch):
    _patch(monkeypatch, _soft({"gsm": "GSM1", "title": "B1"}, {"gsm": "GSM2", "title": "B2"}))
    with pytest.raises(ValueError, match="B9"):
        read_geo("GSE1", samples=["B9"], cache=tmp_path)
    with pytest.raises(ValueError, match="Z9"):
        read_geo("GSE1", groups="B1: control Z9: other", cache=tmp_path)


def test_shared_condition_keeps_the_title_and_says_so(tmp_path, monkeypatch, capsys):
    text = _soft(
        {"gsm": "GSM1", "title": "B1", "chars": {"treatment": "C&AIA"}},
        {"gsm": "GSM2", "title": "B2", "chars": {"treatment": "C&AIA"}},
    )
    _patch(monkeypatch, text)
    _cache_matrices(tmp_path, "GSM1", "GSM2")
    adata = read_geo("GSE1", cache=tmp_path)
    assert "share one recorded condition" in capsys.readouterr().out
    assert set(adata.obs["group"].astype(str)) == {"B1", "B2"}
    assert list(adata.obs["treatment"]) == ["C&AIA"] * 4


def test_titles_and_timepoint_are_read_as_groups(tmp_path, monkeypatch, capsys):
    titled = _soft(
        {"gsm": "GSM1", "title": "Ctrl"},
        {"gsm": "GSM2", "title": "D7"},
    )
    _patch(monkeypatch, titled)
    _cache_matrices(tmp_path, "GSM1", "GSM2")
    adata = read_geo("GSE1", cache=tmp_path)
    assert "share one recorded condition" not in capsys.readouterr().out
    assert set(adata.obs["group"].astype(str)) == {"control", "day 7"}

    timed = _soft(
        {"gsm": "GSM1", "title": "S1", "chars": {"timepoint": "D0", "strain": "C57BL/6"}},
        {"gsm": "GSM2", "title": "S2", "chars": {"timepoint": "D7", "strain": "C57BL/6"}},
    )
    _patch(monkeypatch, timed)
    adata = read_geo("GSE1", samples=["S2"], groups={"GSM2": "day 7 treated"}, cache=tmp_path)
    assert list(adata.obs["sample"].astype(str).unique()) == ["S2"]
    assert list(adata.obs["group"].astype(str).unique()) == ["day 7 treated"]


def test_missing_matrix_and_repeated_title_raise(tmp_path, monkeypatch):
    _patch(monkeypatch, _soft({"gsm": "GSM1", "title": "B1", "files": ("notes.txt",)}))
    with pytest.raises(FileNotFoundError, match="GSM1"):
        read_geo("GSE1", cache=tmp_path)
    _patch(
        monkeypatch,
        _soft({"gsm": "GSM1", "title": "B1"}, {"gsm": "GSM2", "title": "B1"}),
    )
    with pytest.raises(ValueError, match="B1"):
        geo_info("GSE1", cache=tmp_path)


def test_uncompressed_matrix_and_spaced_title(tmp_path, monkeypatch):
    text = _soft(
        {
            "gsm": "GSM1",
            "title": "day 7 treated",
            "files": ("barcodes.tsv", "features.tsv", "matrix.mtx"),
        }
    )
    _patch(monkeypatch, text)
    folder = tmp_path / "GSE1" / "GSM1"
    folder.mkdir(parents=True)
    io.mmwrite(folder / "matrix.mtx", sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 3.0]])).T)
    (folder / "barcodes.tsv").write_text("c1\nc2\n")
    (folder / "features.tsv").write_text(
        "id0\tGeneA\tGene Expression\nid1\tGeneB\tGene Expression\n"
    )
    adata = read_geo(
        "GSE1",
        samples="day 7 treated",
        groups="day 7 treated: control",
        cache=tmp_path,
    )
    assert adata.n_obs == 2
    assert set(adata.obs["group"].astype(str)) == {"control"}


def test_several_h5_matrices_require_one_filtered_file(tmp_path, monkeypatch):
    _patch(
        monkeypatch,
        _soft(
            {
                "gsm": "GSM1",
                "title": "B1",
                "files": ("raw_feature_bc_matrix.h5", "counts.h5"),
            }
        ),
    )
    with pytest.raises(ValueError, match="more than one h5"):
        read_geo("GSE1", cache=tmp_path)
    text = _soft(
        {
            "gsm": "GSM1",
            "title": "B1",
            "files": ("raw_feature_bc_matrix.h5", "filtered_feature_bc_matrix.h5"),
        }
    )
    _patch(monkeypatch, text)
    seen: list[str] = []

    def fake_download(url, dest):
        seen.append(url)
        dest.write_bytes(b"h5")

    monkeypatch.setattr("annotuneagent.geo._download", fake_download)
    monkeypatch.setattr(
        "annotuneagent.geo._read_folder",
        lambda folder: __import__("anndata").AnnData(np.zeros((1, 1))),
    )
    read_geo("GSE1", cache=tmp_path)
    assert seen == ["https://example.test/GSM1_filtered_feature_bc_matrix.h5"]


def test_group_text_with_spaces_is_kept(tmp_path, monkeypatch):
    _patch(monkeypatch, _soft({"gsm": "GSM1", "title": "B1"}, {"gsm": "GSM2", "title": "B2"}))
    _cache_matrices(tmp_path, "GSM1", "GSM2")
    adata = read_geo("GSE1", groups="B1: control group B2: Day 7", cache=tmp_path)
    assert adata.uns["geo"]["groups"] == {"B1": "control group", "B2": "Day 7"}
