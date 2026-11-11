# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Regression tests for the issues found in the 0.1.2 review (2026-11-11)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import annotuneagent as ata
import annotuneagent.local_lora as ll
from annotuneagent import Hypothesis
from annotuneagent.decide import Decision, MlResult


@pytest.fixture(autouse=True)
def _fresh_adapter_ids():
    cache = getattr(ll, "_ADAPTER_IDS", {})
    cache.clear()
    yield
    cache.clear()


def _low_detection_adata() -> AnnData:
    n = 60
    half = n // 2
    x = np.zeros((n, 3))
    x[: int(half * 0.1), 0] = 2.0  # G1: enriched, but in 10% of cluster 0
    x[:half, 1] = 3.0  # S1: a support gene, enriched in every cell of cluster 0
    x[half:, 2] = 3.0
    adata = AnnData(X=x, obs={"leiden": ["0"] * half + ["1"] * half})
    adata.var_names = ["G1", "S1", "G2"]
    adata.uns["log1p"] = {"base": None}
    return adata


def test_positive_gene_below_detection_does_not_unlock_a_label():
    records = ata.annotate(
        _low_detection_adata(), "leiden", [Hypothesis("A", positive=("G1",), support=("S1",))]
    )
    first = next(record for record in records if record["cluster"] == "0")
    assert first["evidence"]["positive"][0]["pct_in"] < 0.25
    assert first["status"] == "uncertain"
    assert first["cell_type"] is None
    assert "G1 enriched" not in first["reasoning"]


def test_hypothesis_string_is_one_gene():
    item = Hypothesis("T", positive="CD3E", negative="CD19", support="", requires_any="CD3D")
    assert item.positive == ("CD3E",)
    assert item.negative == ("CD19",)
    assert item.support == ()
    assert item.requires_any == ("CD3D",)


def test_url_endpoint_uses_its_system_prompt(monkeypatch):
    from annotuneagent.decide import SYSTEM_NAME, fetch_name
    from annotuneagent.workflow import _invoke, _parse_name

    seen = {}

    def complete_url(url, task, system, payload, **_kw):
        seen["system"] = system
        return '{"action": "escalate"}'

    monkeypatch.setattr(ll, "complete_url", complete_url)
    endpoint = ata.TaskEndpoint(url="http://x", system="CUSTOM")
    _invoke(endpoint, {"dictionary_hits": []}, fetch_name, _parse_name, SYSTEM_NAME, "name")
    assert seen["system"] == "CUSTOM"
    _invoke(ata.TaskEndpoint(url="http://x"), {}, fetch_name, _parse_name, SYSTEM_NAME, "name")
    assert seen["system"] == SYSTEM_NAME


def test_cli_hypotheses_file_accepts_bom_strings_and_support(tmp_path: Path):
    from annotuneagent.cli import _hypotheses_from_file

    path = tmp_path / "h.json"
    text = json.dumps(
        [{"name": "γδ T", "positive": "TRDC", "support": ["CD3E"], "requires_any": ["CD3D"]}],
        ensure_ascii=False,
    )
    path.write_bytes(("\ufeff" + text).encode("utf-8"))
    (item,) = _hypotheses_from_file(path)
    assert item.name == "γδ T"
    assert item.positive == ("TRDC",)
    assert item.support == ("CD3E",)
    assert item.requires_any == ("CD3D",)
    path.write_text(json.dumps([{"name": "x", "positive": [1, 2]}]), encoding="utf-8")
    with pytest.raises(ValueError):
        _hypotheses_from_file(path)


def test_report_shows_an_ml_escalation():
    from annotuneagent.workflow import _model_name, _proposal_name

    record = {
        "rounds": [
            {"step": "name", "action": "escalate", "names": []},
            {"step": "ml", "status": "escalate", "names": [], "error": None},
        ]
    }
    assert _model_name(record, "ml") == "escalate"
    assert _proposal_name(record, "ml") is None


def test_repeated_lora_flags_are_read(monkeypatch):
    argv = [
        "/opt/llama-server",
        "-m",
        "/m/base.gguf",
        "--lora",
        "/m/name.gguf",
        "--lora=/m/ml.gguf",
        "--ctx-size=8192",
        "--port",
        "8080",
    ]

    class Entry:
        name = "4242"

    class Proc:
        def is_dir(self):
            return True

        def iterdir(self):
            return [Entry()]

    real_path = ll.Path
    monkeypatch.setattr(ll, "Path", lambda p: Proc() if p == "/proc" else real_path(p))
    monkeypatch.setattr(ll, "_cmdline", lambda pid: argv)
    monkeypatch.setattr(ll, "_listen_port", lambda pid: 8080)
    monkeypatch.setattr(ll, "_health_ok", lambda url: True)
    (found,) = ll._loaded_servers()
    assert found["adapters"] == {"name": "/m/name.gguf", "ml": "/m/ml.gguf"}
    assert found["ctx"] == 8192


def test_geo_keeps_uncompressed_features():
    from annotuneagent.geo import _stored_name

    assert _stored_name("GSM1_features.tsv") == "features.tsv"
    assert _stored_name("GSM1_genes.tsv") == "features.tsv"
    assert _stored_name("GSM1_features.tsv.gz") == "features.tsv.gz"


def test_api_key_is_sent(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"content": "x"}'

    def urlopen(request, timeout=None):
        seen["auth"] = request.get_header("Authorization")
        return Response()

    monkeypatch.setattr(ll.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("ANNOTUNE_API_KEY", "secret")
    ll._post_url("http://h:8080", "/completion", {})
    assert seen["auth"] == "Bearer secret"
    monkeypatch.delenv("ANNOTUNE_API_KEY")
    ll._post_url("http://h:8080", "/completion", {})
    assert seen["auth"] is None


@pytest.mark.parametrize(
    "listed, name_id",
    [
        # The listing MyCloud's llama-server returned on 2026-11-11.
        (
            [
                {"id": 0, "path": "/d/lora/name/annotune-v9h-name-checkpoint-412.lora.f16.gguf"},
                {"id": 1, "path": "/d/lora/ml/annotune-v9e-ml-checkpoint-231.lora.f16.gguf"},
            ],
            0,
        ),
        (
            [
                {"id": 0, "path": "C:\\w\\annotune-v9e-ml-checkpoint-231.lora.f16.gguf"},
                {"id": 1, "path": "C:\\w\\annotune-v9h-name-checkpoint-412.lora.f16.gguf"},
            ],
            1,
        ),
        ([{"id": 0, "path": "/a.gguf"}, {"id": 1, "path": "/b.gguf"}], 0),
        (None, 0),
    ],
)
def test_url_requests_enable_the_adapter_by_file_name(monkeypatch, listed, name_id):
    seen = {}
    monkeypatch.setattr(ll, "_get_json", lambda url, path: listed)
    monkeypatch.setattr(ll, "render_prompt", lambda *a, **k: "P")
    monkeypatch.setattr(ll, "prompt_tokens", lambda *a, **k: 4)

    def post(url, path, payload, timeout=None):
        seen["url"] = f"{url}{path}"
        seen["lora"] = payload["lora"]
        return {"content": "{}"}

    monkeypatch.setattr(ll, "_post_url", post)
    ll.complete_url("http://h:8080/", "name", "s", "u", maxlen=100)
    assert seen["url"] == "http://h:8080/completion"
    enabled = [item["id"] for item in seen["lora"] if item["scale"] == 1.0]
    assert enabled == [name_id]
    assert sorted(item["id"] for item in seen["lora"]) == [0, 1]


def test_evaluate_tolerates_clusters_without_a_record():
    adata = _low_detection_adata()
    adata.obs["author"] = ["x"] * adata.n_obs
    records = ata.annotate(adata, "leiden", [Hypothesis("A", positive=("S1",))], clusters=["0"])
    result = ata.evaluate(records, adata, "leiden", "author")
    assert result["n_clusters"] == 2
    assert result["n_uncertain"] >= 1


def _three_clusters(sizes=(40, 49, 1)) -> AnnData:
    rng = np.random.default_rng(1)
    n = sum(sizes)
    x = rng.poisson(0.3, (n, 4)).astype(float)
    labels: list[str] = []
    for index, size in enumerate(sizes):
        labels += [str(index)] * size
    x[: sizes[0], 0] += 4
    x[sizes[0] : sizes[0] + sizes[1], 1] += 4
    adata = AnnData(X=np.log1p(x), obs={"leiden": labels})
    adata.var_names = ["A1", "B1", "C1", "Z0"]
    adata.uns["log1p"] = {"base": None}
    return adata


_HYPOTHESES = [
    Hypothesis("TA", positive=("A1",), negative=("B1",)),
    Hypothesis("TB", positive=("B1",), negative=("A1",)),
]


def _ask_models():
    return ata.ModelChain(
        name=ata.TaskEndpoint(ask=lambda payload: Decision("escalate")),
        ml=ata.TaskEndpoint(ask=lambda payload: MlResult("escalate")),
    )


@pytest.mark.parametrize("kind", ["singleton", "integer", "unused_category"])
def test_models_run_with_awkward_cluster_columns(kind):
    adata = _three_clusters()
    if kind == "integer":
        adata.obs["leiden"] = adata.obs["leiden"].astype(int)
    if kind == "unused_category":
        adata.obs["leiden"] = pd.Categorical(adata.obs["leiden"], categories=["0", "1", "2", "9"])
    report = ata.run(
        adata, "leiden", hypotheses=_HYPOTHESES, subcluster=False, models=_ask_models()
    )
    labels = {row["cluster"]: row["label"] for row in report["clusters"]}
    assert labels["0"] == "TA"
    assert labels["1"] == "TB"
    assert "_annotune_wilcoxon_groups" not in adata.obs.columns


def test_integer_cluster_column_without_models():
    # Cluster 2 has no marker, so the Wilcoxon vote runs for it.
    adata = _three_clusters(sizes=(40, 40, 10))
    adata.obs["leiden"] = adata.obs["leiden"].astype(int)
    report = ata.run(adata, "leiden", hypotheses=_HYPOTHESES, subcluster=False)
    labels = {row["cluster"]: row["label"] for row in report["clusters"]}
    assert labels["0"] == "TA" and labels["1"] == "TB"


def test_missing_tokenizer_stack_fails_before_scoring(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "jinja2" else real(name, *a)
    )
    models = ata.ModelChain(
        name=ata.TaskEndpoint(url="http://h:8080"), ml=ata.TaskEndpoint(url="http://h:8080")
    )
    with pytest.raises(ImportError, match="annotuneagent\\[models\\]"):
        ata.run(_three_clusters(), "leiden", hypotheses=_HYPOTHESES, models=models)
    # ask callbacks need no tokenizer
    ata.run(
        _three_clusters(), "leiden", hypotheses=_HYPOTHESES, models=_ask_models(), subcluster=False
    )


def test_write_h5ad_keeps_the_prepare_report(tmp_path: Path):
    import anndata as ad

    adata = _three_clusters()
    adata.uns["annotune_prep"] = {"qc": {"batches": [{"batch": "all", "rules": [{"n": 1}]}]}}
    with pytest.raises(Exception):
        adata.copy().write_h5ad(tmp_path / "plain.h5ad")
    ata.write_h5ad(adata, tmp_path / "ok.h5ad")
    back = ad.read_h5ad(tmp_path / "ok.h5ad")
    assert json.loads(back.uns["annotune_prep"])["qc"]["batches"][0]["batch"] == "all"
    assert isinstance(adata.uns["annotune_prep"], dict)


def test_models_extra_includes_jinja2():
    tomllib = pytest.importorskip("tomllib")

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    extra = " ".join(project["optional-dependencies"]["models"])
    assert "transformers" in extra and "jinja2" in extra
