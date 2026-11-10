import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from anndata import AnnData

from annotuneagent.cli import main
from annotuneagent.decide import Decision


def test_cli_can_select_solo_and_scvi_independently(tmp_path: Path, monkeypatch):
    from importlib import import_module

    prep_module = import_module("annotuneagent.prepare")

    rng = np.random.default_rng(4)
    x = rng.poisson(2, size=(60, 40)).astype(float)
    x[:30, :10] += 15
    x[30:, 10:20] += 15
    adata = AnnData(X=x, obs={"sample": ["a"] * 30 + ["b"] * 30})
    adata.var_names = [f"Gene{i}" for i in range(40)]
    adata.obs_names = [f"cell{i}" for i in range(60)]
    src = tmp_path / "input.h5ad"
    out = tmp_path / "report.json"
    labeled = tmp_path / "prepared.h5ad"
    hypotheses = tmp_path / "hypotheses.json"
    rates = tmp_path / "rates.json"
    with ad.settings.override(allow_write_nullable_strings=True):
        adata.write_h5ad(src)
    hypotheses.write_text(json.dumps([{"name": "candidate", "positive": ["Gene0"]}]))
    rates.write_text(json.dumps({"a": 0.1, "b": 0.2}))
    seen = []

    def solo(data, batch_key, expected_rates, chip):
        seen.append(("solo", batch_key, expected_rates, chip))
        data.obs["solo_score"] = 0.1
        data.obs["solo_prediction"] = "singlet"
        data.obs["is_doublet"] = False
        return {"enabled": True, "method": "solo", "n_removed": 0, "batches": []}

    def scvi(data, batch_key, n_latent, model_dir):
        seen.append(("scvi", batch_key, n_latent, model_dir))
        latent = np.random.default_rng(5).normal(size=(data.n_obs, n_latent))
        latent[data.obs["sample"].to_numpy() == "b", :5] += 12
        data.obsm["X_scVI"] = latent

    monkeypatch.setattr(prep_module, "_drop_solo_doublets", solo)
    monkeypatch.setattr(prep_module, "_fit_scvi", scvi)
    # The stubs stand in for scVI. The install check must not return first.
    monkeypatch.setattr(prep_module, "_is_missing", lambda name: False)
    assert (
        main(
            [
                "--input",
                str(src),
                "--output",
                str(out),
                "--output-h5ad",
                str(labeled),
                "--hypotheses",
                str(hypotheses),
                "--species",
                "human",
                "--prepare",
                "--batch-key",
                "sample",
                "--batch-correction",
                "scvi",
                "--doublet-method",
                "solo",
                "--solo-expected-rates",
                str(rates),
            ]
        )
        == 0
    )
    assert seen == [("solo", "sample", {"a": 0.1, "b": 0.2}, "10k"), ("scvi", "sample", 30, None)]
    saved = ad.read_h5ad(labeled)
    assert "X_scVI" in saved.obsm and "leiden" in saved.obs
    assert json.loads(saved.uns["annotune_prep"])["doublets"]["method"] == "solo"
    assert out.is_file()


def test_cli_refuses_to_overwrite_an_input(tmp_path: Path):
    src = tmp_path / "cells.h5ad"
    src.write_bytes(b"counts")
    hypotheses = tmp_path / "hypotheses.json"
    hypotheses.write_text("[]")
    with pytest.raises(SystemExit):
        main(
            [
                "--input",
                str(src),
                "--output",
                str(src),
                "--hypotheses",
                str(hypotheses),
                "--cluster-key",
                "leiden",
            ]
        )
    assert src.read_bytes() == b"counts"
    report = tmp_path / "report.json"
    with pytest.raises(SystemExit):
        main(
            [
                "--input",
                str(src),
                "--output",
                str(report),
                "--output-h5ad",
                str(report),
                "--hypotheses",
                str(hypotheses),
                "--cluster-key",
                "leiden",
            ]
        )


def test_cli_prepare_prints_install_line_when_scvi_is_missing(tmp_path: Path, monkeypatch, capsys):
    from importlib import import_module

    prep_module = import_module("annotuneagent.prepare")
    monkeypatch.setattr(prep_module, "_is_missing", lambda name: name == "scvi")
    src = tmp_path / "input.h5ad"
    out = tmp_path / "report.json"
    hypotheses = tmp_path / "hypotheses.json"
    with ad.settings.override(allow_write_nullable_strings=True):
        AnnData(X=np.ones((8, 4))).write_h5ad(src)
    hypotheses.write_text(json.dumps([{"name": "candidate", "positive": ["Gene0"]}]))
    assert (
        main(
            [
                "--input",
                str(src),
                "--output",
                str(out),
                "--hypotheses",
                str(hypotheses),
                "--species",
                "human",
                "--prepare",
                "--batch-correction",
                "scvi",
            ]
        )
        == 0
    )
    assert "scvi is not installed. Install it with:" in capsys.readouterr().out
    assert not out.exists()


@pytest.mark.parametrize(
    "scope, cell_type, positives",
    [
        ("adipose", "adipocyte", ["ADIPOQ", "PLIN1", "FABP4"]),
        ("tendon", "tenocyte", ["SCX", "MKX", "TNMD"]),
        ("stomach", "foveolar cell", ["MUC5AC", "GKN1", "TFF1"]),
        ("cartilage", "chondrocyte", ["COL2A1", "ACAN", "SOX9"]),
        ("thyroid", "thyroid follicular cell", ["TG", "TPO"]),
        ("bladder", "bladder urothelial cell", ["UPK1B", "UPK2"]),
        ("salivary", "serous acinar cell of salivary gland", ["AMY1A", "STATH"]),
        ("retina", "retinal rod cell", ["RHO", "NRL"]),
        ("testis", "Sertoli cell", ["SOX9", "CLDN11"]),
        ("ovary", "granulosa cell", ["FOXL2", "INHA"]),
        ("mammary", "luminal epithelial cell of mammary gland", ["KRT8", "KRT18"]),
        ("prostate", "luminal cell of prostate epithelium", ["KLK3", "ACPP"]),
        ("endometrium", "glandular epithelial cell of endometrium", ["FOXA2", "EPCAM"]),
        ("cornea", "corneal epithelial cell", ["KRT3", "KRT12"]),
        ("adrenal", "cortical cell of adrenal gland", ["STAR", "CYP11A1"]),
        ("placenta", "extravillous trophoblast", ["HLA-G", "MMP2"]),
        ("thymus", "epithelial cell of thymus", ["FOXN1", "EPCAM"]),
        ("intestine", "enteric neuron", ["ELAVL4", "PHOX2B"]),
        ("heart", "mesothelial cell of epicardium", ["UPK3B", "MSLN"]),
        ("intestine", "interstitial cell of Cajal", ["KIT", "ANO1"]),
        ("kidney", "parietal epithelial cell", ["CLDN1", "CFH", "ALDH1A2"]),
    ],
)
def test_new_tissue_page_is_reachable_from_cli(tmp_path: Path, scope, cell_type, positives):
    genes = positives + ["PTPRC", "PECAM1"]
    x = np.zeros((20, len(genes)), dtype=float)
    x[:10, : len(positives)] = 5
    x[10:, len(positives)] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 10 + ["1"] * 10})
    adata.var_names = genes
    adata.uns["log1p"] = {"base": None}
    src = tmp_path / "input.h5ad"
    dest = tmp_path / "out.json"
    with ad.settings.override(allow_write_nullable_strings=True):
        adata.write_h5ad(src)
    assert (
        main(
            [
                "--input",
                str(src),
                "--cluster-key",
                "leiden",
                "--species",
                "human",
                "--tissue",
                scope,
                "--types",
                cell_type,
                "--output",
                str(dest),
            ]
        )
        == 0
    )
    payload = json.loads(dest.read_text())
    assert payload["records"][0]["cell_type"] == cell_type


def test_cli_writes_records_and_evaluation(tmp_path: Path):
    genes = ["CD3E", "CD4", "IL7R", "CD8A", "CD8B", "MS4A1"]
    x = np.zeros((40, len(genes)), dtype=float)
    for gene in ("CD3E", "CD4", "IL7R"):
        x[:20, genes.index(gene)] = 5
    for gene in ("CD3E", "CD8A", "CD8B"):
        x[20:, genes.index(gene)] = 5
    adata = AnnData(
        X=x,
        obs={
            "leiden": ["0"] * 20 + ["1"] * 20,
            "author": ["CD4-positive, alpha-beta T cell"] * 20
            + ["CD8-positive, alpha-beta T cell"] * 20,
        },
    )
    adata.var_names = genes
    adata.uns["log1p"] = {"base": None}
    src = tmp_path / "in.h5ad"
    dest = tmp_path / "out.json"
    with ad.settings.override(allow_write_nullable_strings=True):
        adata.write_h5ad(src)
    code = main(
        [
            "--input",
            str(src),
            "--cluster-key",
            "leiden",
            "--species",
            "human",
            "--types",
            "CD4-positive, alpha-beta T cell",
            "--types",
            "CD8-positive, alpha-beta T cell",
            "--label-key",
            "author",
            "--output",
            str(dest),
        ]
    )
    assert code == 0
    payload = json.loads(dest.read_text())
    assert payload["evaluation"]["n_exact"] == 2
    assert payload["dotplot"] == str(tmp_path / "out.markers.pdf")
    assert (tmp_path / "out.markers.pdf").is_file()
    assert payload["marker_dict"]
    assert "fine_order" in payload["annotation_code"]
    assert {row["cell_type"] for row in payload["records"]} == {
        "CD4-positive, alpha-beta T cell",
        "CD8-positive, alpha-beta T cell",
    }


def test_list_tissues_for_an_author_source(capsys):
    assert main(["--list-tissues", "--author-tissue", "DRG", "--species", "mouse"]) == 0
    assert capsys.readouterr().out.strip().splitlines() == ["drg"]


def test_cli_rejects_unknown_tissue_before_reading_input(capsys, tmp_path: Path):
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--input",
                str(tmp_path / "missing.h5ad"),
                "--output",
                str(tmp_path / "out.json"),
                "--hypotheses",
                str(tmp_path / "missing.json"),
                "--species",
                "human",
                "--tissue",
                "stmoach",
            ]
        )
    assert "unknown human tissue 'stmoach'" in capsys.readouterr().err


@pytest.mark.parametrize(
    "option",
    [
        ("--doublet-method", "solo"),
        ("--batch-correction", "scvi"),
        ("--batch-key", "sample"),
        ("--solo-expected-rates", "rates.json"),
        ("--n-pcs", "10"),
    ],
)
def test_cli_rejects_prepare_options_without_prepare(option, capsys, tmp_path: Path):
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--input",
                str(tmp_path / "missing.h5ad"),
                "--output",
                str(tmp_path / "out.json"),
                "--hypotheses",
                str(tmp_path / "missing.json"),
                *option,
            ]
        )
    assert f"{option[0]} require --prepare" in capsys.readouterr().err


def test_cli_creates_report_and_h5ad_directories_without_marker_plot(tmp_path: Path):
    adata = AnnData(
        X=np.array([[2.3, 0.0]] * 10 + [[0.0, 2.3]] * 10), obs={"leiden": ["0"] * 10 + ["1"] * 10}
    )
    adata.var_names = ["G1", "G2"]
    adata.uns["log1p"] = {"base": None}
    src = tmp_path / "input.h5ad"
    hypotheses = tmp_path / "hypotheses.json"
    report = tmp_path / "reports" / "result.json"
    labeled = tmp_path / "labeled" / "result.h5ad"
    adata.write_h5ad(src)
    hypotheses.write_text(json.dumps([{"name": "type", "positive": ["G1"]}]))

    assert (
        main(
            [
                "--input",
                str(src),
                "--output",
                str(report),
                "--output-h5ad",
                str(labeled),
                "--cluster-key",
                "leiden",
                "--hypotheses",
                str(hypotheses),
            ]
        )
        == 0
    )
    assert report.is_file() and labeled.is_file()
    assert json.loads(report.read_text())["dotplot"] is None


def test_extra_gene_is_listed_and_not_a_label(tmp_path: Path):
    x = np.zeros((40, 4), dtype=float)
    x[:20, 0] = 5
    x[:20, 1] = 5
    x[:20, 3] = 8
    x[20:, 0] = 5
    x[20:, 2] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20})
    adata.var_names = ["CD3E", "CD4", "CD8A", "IL7R"]
    adata.uns["log1p"] = {"base": None}
    src = tmp_path / "in.h5ad"
    dest = tmp_path / "out.json"
    hypotheses = tmp_path / "h.json"
    with ad.settings.override(allow_write_nullable_strings=True):
        adata.write_h5ad(src)
    hypotheses.write_text(
        json.dumps(
            [
                {"name": "CD4 T", "positive": ["CD3E", "CD4"], "negative": ["CD8A"]},
                {"name": "CD8 T", "positive": ["CD3E", "CD8A"], "negative": ["CD4"]},
            ]
        )
    )
    main(
        [
            "--input",
            str(src),
            "--cluster-key",
            "leiden",
            "--hypotheses",
            str(hypotheses),
            "--output",
            str(dest),
        ]
    )
    records = json.loads(dest.read_text())["records"]
    cluster0 = next(row for row in records if row["cluster"] == "0")
    assert cluster0["cell_type"] == "CD4 T"
    assert "IL7R" in cluster0["evidence"]["de_not_in_markers"]


def test_cli_workflow_saves_labels(tmp_path: Path):
    x = np.zeros((20, 2), dtype=float)
    x[:10, 0] = 5
    x[10:, 1] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 10 + ["1"] * 10})
    adata.var_names = ["A", "B"]
    adata.uns["log1p"] = {"base": None}
    src = tmp_path / "in.h5ad"
    dest = tmp_path / "out.json"
    labeled = tmp_path / "labeled.h5ad"
    hypotheses = tmp_path / "h.json"
    with ad.settings.override(allow_write_nullable_strings=True):
        adata.write_h5ad(src)
    hypotheses.write_text(
        json.dumps(
            [
                {"name": "A", "positive": ["A"]},
                {"name": "B", "positive": ["B"]},
            ]
        )
    )
    assert (
        main(
            [
                "--input",
                str(src),
                "--cluster-key",
                "leiden",
                "--hypotheses",
                str(hypotheses),
                "--output",
                str(dest),
                "--output-h5ad",
                str(labeled),
            ]
        )
        == 0
    )
    assert json.loads(dest.read_text())["all_labeled"] is True
    assert list(ad.read_h5ad(labeled).obs["annotune_label"].astype(str)) == ["A"] * 10 + ["B"] * 10


def test_cli_local_models_load_only_name_and_ml(tmp_path: Path, monkeypatch):
    x = np.zeros((40, 2), dtype=float)
    x[:20, 0] = 5
    x[20:, 1] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20})
    adata.var_names = ["A", "B"]
    adata.uns["log1p"] = {"base": None}
    src = tmp_path / "in.h5ad"
    out = tmp_path / "out.json"
    labeled = tmp_path / "labeled.h5ad"
    hypotheses = tmp_path / "hypotheses.json"
    adata.write_h5ad(src)
    hypotheses.write_text(json.dumps([{"name": "A cell", "positive": ["A"]}]))
    seen = []

    def open_two(adapters, *, gguf, tokenizer, url=None):
        seen.append((adapters, gguf, tokenizer, url))

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("annotuneagent.local_lora.open_adapters", open_two)
    monkeypatch.setattr("annotuneagent.local_lora.complete", unavailable)
    assert (
        main(
            [
                "--input",
                str(src),
                "--cluster-key",
                "leiden",
                "--species",
                "human",
                "--hypotheses",
                str(hypotheses),
                "--output",
                str(out),
                "--output-h5ad",
                str(labeled),
                "--base",
                "base.gguf",
                "--name-adapter",
                "name.gguf",
                "--ml-adapter",
                "ml.gguf",
            ]
        )
        == 0
    )
    assert len(seen) == 1
    assert seen[0][0] == {"name": "name.gguf", "ml": "ml.gguf"}
    assert out.is_file() and labeled.is_file()


def test_cli_model_nominates_a_dictionary_type_and_writes_it(tmp_path: Path, monkeypatch):
    genes = ["CD3E", "CD4", "IL7R", "CD8A"]
    x = np.zeros((40, len(genes)), dtype=float)
    for gene in ("CD3E", "CD4", "IL7R"):
        x[:20, genes.index(gene)] = 5
    x[20:, genes.index("CD8A")] = 5
    adata = AnnData(
        X=x,
        obs={"leiden": ["0"] * 20 + ["1"] * 20, "author": ["CD4 T"] * 40},
    )
    adata.var_names = genes
    adata.uns["log1p"] = {"base": None}
    src = tmp_path / "in.h5ad"
    dest = tmp_path / "out.json"
    labeled = tmp_path / "labeled.h5ad"
    hypotheses = tmp_path / "scored.json"
    catalog = tmp_path / "catalog.json"
    with ad.settings.override(allow_write_nullable_strings=True):
        adata.write_h5ad(src)
    hypotheses.write_text(
        json.dumps([{"name": "broad T", "positive": ["CD3E", "CD4", "IL7R"], "negative": ["CD4"]}])
    )
    catalog.write_text(
        json.dumps(
            [
                {"name": "broad T", "positive": ["CD3E", "CD4", "IL7R"], "negative": ["CD4"]},
                {"name": "CD4 T", "positive": ["CD3E", "CD4", "IL7R"], "negative": ["CD8A"]},
            ]
        )
    )
    seen = []

    def nominate_name(_url, payload, **_extra):
        seen.append(payload)
        return Decision("retest", ("CD4 T",))

    def nominate_ml(_url, payload, **_extra):
        assert "marker" not in payload
        from annotuneagent.decide import MlResult

        return MlResult("escalate")

    monkeypatch.setattr("annotuneagent.workflow.fetch_name", nominate_name)
    monkeypatch.setattr("annotuneagent.workflow.fetch_ml", nominate_ml)
    assert (
        main(
            [
                "--input",
                str(src),
                "--cluster-key",
                "leiden",
                "--hypotheses",
                str(hypotheses),
                "--catalog",
                str(catalog),
                "--output",
                str(dest),
                "--output-h5ad",
                str(labeled),
                "--label-key",
                "author",
                "--name-url",
                "http://127.0.0.1:10",
                "--ml-url",
                "http://127.0.0.1:11",
            ]
        )
        == 0
    )
    labels = list(ad.read_h5ad(labeled).obs["annotune_label"].astype(str))
    assert labels[:20] == ["CD4 T"] * 20
    assert seen and "dictionary_hits" in seen[0]
    output = json.loads(dest.read_text())
    rounds = output["records"]
    cluster0 = next(row for row in rounds if row["cluster"] == "0")
    assert cluster0["added"] == ["CD4 T"]
    assert [step["step"] for step in cluster0["rounds"]] == ["name", "ml"]
    evaluated = next(row for row in output["evaluation"]["clusters"] if row["cluster"] == "0")
    assert evaluated["predicted"] == "CD4 T"
    assert evaluated["exact"] is True


def test_cli_model_path_saves_unresolved_cells(tmp_path: Path, monkeypatch):
    adata = AnnData(X=np.zeros((20, 1)), obs={"leiden": ["0"] * 10 + ["1"] * 10})
    adata.var_names = ["A"]
    adata.uns["log1p"] = {"base": None}
    src = tmp_path / "in.h5ad"
    dest = tmp_path / "out.json"
    labeled = tmp_path / "labeled.h5ad"
    hypotheses = tmp_path / "h.json"
    with ad.settings.override(allow_write_nullable_strings=True):
        adata.write_h5ad(src)
    hypotheses.write_text(json.dumps([{"name": "A", "positive": ["A"]}]))
    calls = []

    def no_name(url, _payload, **_extra):
        calls.append("name:" + url)
        return Decision("retest", ("A",))

    def ml(url, _payload, **_extra):
        calls.append(url)
        from annotuneagent.decide import MlResult

        return MlResult("escalate")

    monkeypatch.setattr("annotuneagent.workflow.fetch_name", no_name)
    monkeypatch.setattr("annotuneagent.workflow.fetch_ml", ml)
    assert (
        main(
            [
                "--input",
                str(src),
                "--cluster-key",
                "leiden",
                "--hypotheses",
                str(hypotheses),
                "--output",
                str(dest),
                "--output-h5ad",
                str(labeled),
                "--name-url",
                "http://localhost:9932",
                "--ml-url",
                "http://localhost:9933",
            ]
        )
        == 0
    )
    assert calls == ["name:http://localhost:9932", "http://localhost:9933"] * 2
    assert ad.read_h5ad(labeled).obs["annotune_label"].isna().all()
    assert json.loads(dest.read_text())["queue"] == ["0", "1"]
    report = json.loads(dest.read_text())
    assert len(report["records"]) == 2
    assert all(
        row["final_status"] == "review" and row["final_reason"] == "small_cluster"
        for row in report["records"]
    )
    assert set(ad.read_h5ad(labeled).obs["annotune_final_status"]) == {"review"}


def test_cli_writes_a_marker_name_that_name_and_ml_repeat(tmp_path: Path, monkeypatch):
    from annotuneagent.decide import Decision, MlResult

    x = np.zeros((40, 2))
    x[:20, 0] = 5
    x[20:, 1] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20})
    adata.var_names = ["A", "B"]
    adata.uns["log1p"] = {"base": None}
    src = tmp_path / "in.h5ad"
    out = tmp_path / "out.json"
    labeled = tmp_path / "labeled.h5ad"
    hypotheses = tmp_path / "h.json"
    with ad.settings.override(allow_write_nullable_strings=True):
        adata.write_h5ad(src)
    hypotheses.write_text(json.dumps([{"name": "A cell", "positive": ["A"]}]))
    monkeypatch.setattr(
        "annotuneagent.workflow.fetch_name",
        lambda _url, _payload, **_extra: Decision("retest", ("A cell",)),
    )
    monkeypatch.setattr(
        "annotuneagent.workflow.fetch_ml",
        lambda _url, _payload, **_extra: MlResult("named", ("A cell",)),
    )
    assert (
        main(
            [
                "--input",
                str(src),
                "--cluster-key",
                "leiden",
                "--hypotheses",
                str(hypotheses),
                "--output",
                str(out),
                "--output-h5ad",
                str(labeled),
                "--name-url",
                "http://localhost:9932",
                "--ml-url",
                "http://localhost:9933",
            ]
        )
        == 0
    )
    record = next(row for row in json.loads(out.read_text())["records"] if row["cluster"] == "0")
    assert record["final_status"] == "named"
    assert record["final_reason"] == "evidence_agreement"
    assert record["written_label"] == "A cell"
    assert record["hint"] is None
    saved = ad.read_h5ad(labeled).obs
    assert set(saved["annotune_label"].iloc[:20]) == {"A cell"}
    assert saved["annotune_hint"].iloc[:20].isna().all()
    assert set(saved["annotune_final_status"].iloc[:20]) == {"named"}


def test_cli_stops_before_any_work_when_the_report_cannot_be_written(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "--input",
                str(tmp_path / "never_read.h5ad"),
                "--cluster-key",
                "leiden",
                "--types",
                "T cell",
                "--species",
                "human",
                "--output",
                str(blocker / "report.json"),
            ]
        )
    assert stopped.value.code == 2
    assert "cannot create" in capsys.readouterr().err
