from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from annotuneagent import Hypothesis, evaluate, hypotheses_from_names, local_models, run
from annotuneagent import workflow as wf
from annotuneagent.decide import Decision, MlResult, wilcoxon_order
from annotuneagent.local_lora import scales_for
from annotuneagent.workflow import ModelChain, TaskEndpoint, settle_cluster


def _pure():
    x = np.zeros((40, 4), dtype=float)
    x[:20, 0] = 5
    x[:20, 1] = 5
    x[20:, 0] = 5
    x[20:, 2] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20})
    adata.var_names = ["CD3E", "CD4", "CD8A", "IL7R"]
    adata.uns["log1p"] = {"base": None}
    return adata


def _hypotheses():
    return [
        Hypothesis("CD4 T", positive=("CD3E", "CD4"), negative=("CD8A",)),
        Hypothesis("CD8 T", positive=("CD3E", "CD8A"), negative=("CD4",)),
    ]


def test_stale_subcluster_column_is_ignored(tmp_path):
    x = np.zeros((90, 3), dtype=float)
    x[:30, 0] = 5
    x[30:60, 1] = 5
    x[60:, 2] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 30 + ["1"] * 30 + ["2"] * 30})
    adata.var_names = ["X", "Y", "Z"]
    adata.uns["log1p"] = {"base": None}
    adata.obs["annotune_sub"] = ["9"] * 60 + ["2"] * 30
    report = run(
        adata,
        "leiden",
        hypotheses=[
            Hypothesis("TypeX", positive=("X",)),
            Hypothesis("TypeY", positive=("Y",)),
            Hypothesis("TypeZ", positive=("Z",)),
        ],
        subcluster=False,
    )
    assert "annotune_sub" not in adata.obs.columns
    assert list(adata.obs["annotune_label"].iloc[:30].astype(str).unique()) == ["TypeX"]
    assert list(adata.obs["annotune_label"].iloc[30:60].astype(str).unique()) == ["TypeY"]
    assert "'0': 'TypeX'" in report["annotation_code"]
    assert "'1': 'TypeY'" in report["annotation_code"]
    assert "'2': 'TypeZ'" in report["annotation_code"]
    assert "assert 'leiden' in adata.obs" in report["annotation_code"]
    assert "drop_clusters = []" in report["annotation_code"]
    assert "adata.obs['leiden']" in report["annotation_code"]


def test_duplicate_gene_names_are_rejected():
    adata = AnnData(X=np.zeros((6, 2), dtype=float), obs={"leiden": ["0"] * 3 + ["1"] * 3})
    adata.var_names = ["A", "A"]
    adata.uns["log1p"] = {"base": None}
    try:
        run(adata, "leiden", hypotheses=[Hypothesis("TypeA", positive=("A",))])
    except ValueError as exc:
        assert "not unique" in str(exc)
    else:
        raise AssertionError("duplicate var_names were accepted")


def test_query_lists_twenty_enriched_genes():
    from annotuneagent.workflow import _cluster_query

    genes = [f"G{i}" for i in range(25)]
    x = np.zeros((40, 25), dtype=float)
    x[:20] = 3
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20})
    adata.var_names = genes
    inside = pd.DataFrame({"0": np.full(25, 0.9), "1": np.full(25, 0.1)}, index=genes)
    outside = pd.DataFrame({"0": np.full(25, 0.2), "1": np.full(25, 0.4)}, index=genes)
    adata.uns["rank_genes_groups"] = {
        "params": {"groupby": "leiden"},
        "names": {"0": np.array(genes)},
        "logfoldchanges": {"0": np.array([float(25 - i) for i in range(25)])},
        "pts": inside,
        "pts_rest": outside,
    }
    text = _cluster_query(
        adata, "0", species="mouse", tissue="bone", cells=20, programs=[], by_name={}
    )
    line = next(line for line in text.splitlines() if line.startswith("Enriched"))
    body = line.removeprefix("Enriched versus other cells: ").removesuffix(".")
    listed = [part.split(" ", 1)[0] for part in body.split("; ")]
    assert listed == genes[:20]


def test_query_ignores_a_rank_table_from_another_grouping():
    from annotuneagent.workflow import _attach_queries

    x = np.zeros((40, 1), dtype=float)
    x[:20, 0] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20})
    adata.var_names = ["CD4"]
    adata.uns["log1p"] = {"base": None}
    adata.uns["rank_genes_groups"] = {
        "params": {"groupby": "leiden"},
        "names": {"0": np.array(["FAKE"])},
        "logfoldchanges": {"0": np.array([9.0])},
    }
    record = {"cluster": "0", "written_label": None, "hint": "check", "ranking": []}
    _attach_queries(adata, "leiden", [record], [], None, None)
    assert record["query"] is not None
    assert "CD4" in record["query"]
    assert "FAKE" not in record["query"]


def test_retest_stays_on_the_cluster_that_asked_for_it():
    x = np.zeros((40, 4), dtype=float)
    x[:20, 0] = 5
    x[:20, 1] = 4.9
    x[20:, 2] = 5
    x[20:, 3] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20})
    adata.var_names = ["A", "B", "C", "E"]
    adata.uns["log1p"] = {"base": None}
    seen: list[list[str]] = []

    def name(payload):
        seen.append([item["name"] for item in payload["dictionary_hits"]])
        genes = {item["gene"] for item in payload["genes"]}
        if "A" in genes:
            return Decision("retest", ("TypeExtra",))
        return Decision("escalate")

    def ml(_payload):
        return MlResult("escalate")

    shared = [
        Hypothesis("TypeA", positive=("A",)),
        Hypothesis("TypeB", positive=("B",)),
        Hypothesis("TypeC", positive=("C",)),
        Hypothesis("TypeExtra", positive=("E",)),
    ]
    run(
        adata,
        "leiden",
        hypotheses=shared[:3],
        catalog=shared,
        models=_chain(name, ml),
        subcluster=False,
    )
    assert len(seen) == 2
    assert "TypeExtra" not in seen[1]


def test_open_local_passes_the_callers_tokenizer(monkeypatch):
    seen = {}

    def fake(adapters, *, gguf=None, tokenizer=None, ctx=8192, url=None):
        seen["adapters"] = adapters
        seen["gguf"] = gguf
        seen["tokenizer"] = tokenizer
        seen["ctx"] = ctx
        seen["url"] = url

    monkeypatch.setattr("annotuneagent.local_lora.open_adapters", fake)
    chain = ModelChain(
        name=TaskEndpoint(adapter="name.gguf", base="base.gguf", tokenizer="/tok", maxlen=2048),
        ml=TaskEndpoint(adapter="ml.gguf", base="base.gguf", tokenizer="/tok", maxlen=8192),
    )
    from annotuneagent.workflow import _open_local

    assert _open_local(chain) is True
    assert seen["gguf"] == "base.gguf"
    assert seen["tokenizer"] == "/tok"
    assert seen["adapters"] == {"name": "name.gguf"}
    assert seen["ctx"] == 2048


def test_open_local_rejects_mixed_callback_and_adapter():
    from annotuneagent.workflow import _open_local

    chain = ModelChain(
        name=TaskEndpoint(ask=lambda payload: payload, adapter="name.gguf", base="base.gguf"),
        ml=TaskEndpoint(adapter="ml.gguf", base="base.gguf"),
    )
    with pytest.raises(ValueError, match="cannot be combined with ask callbacks"):
        _open_local(chain)


def test_local_models_use_the_given_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("ANNOTUNE_URL", raising=False)
    root = tmp_path / "weights"
    chain = local_models(root, tokenizer="/tok")
    for endpoint in (chain.name, chain.ml):
        assert endpoint.adapter is not None
        assert Path(endpoint.adapter).is_relative_to(root)
        assert endpoint.base.endswith(".gguf")
        assert Path(endpoint.base).is_relative_to(root)
        assert endpoint.tokenizer == "/tok"
        assert endpoint.server == "http://127.0.0.1:8080"


def test_local_models_use_the_packaged_tokenizer(tmp_path):
    from annotuneagent.local_lora import packaged_tokenizer

    chain = local_models(tmp_path)
    assert chain.name.tokenizer == packaged_tokenizer()
    assert Path(packaged_tokenizer(), "tokenizer.json").is_file()


def test_wilcoxon_cache_recomputes_after_cluster_labels_change():
    adata = _pure()
    first = wilcoxon_order(adata, "leiden")
    adata.obs["leiden"] = ["1"] * 20 + ["0"] * 20
    second = wilcoxon_order(adata, "leiden")
    assert first["0"][0][0] != second["0"][0][0]


def test_wilcoxon_cache_recomputes_after_inplace_expression_edit():
    x = np.zeros((40, 2), dtype=float)
    x[:20, 0] = 5
    x[20:, 1] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20})
    adata.var_names = ["A", "B"]
    first = wilcoxon_order(adata, "leiden")
    assert first["0"][0][0] == "A"
    adata.X[:20] = (0, 5)
    adata.X[20:] = (5, 0)
    second = wilcoxon_order(adata, "leiden")
    assert second["0"][0][0] == "B"


def test_show_prints_the_editable_code(monkeypatch, capsys):
    def no_plot(*args, **kwargs):
        return None

    monkeypatch.setattr("scanpy.pl.umap", no_plot)
    monkeypatch.setattr("annotuneagent.workflow.plot_marker_panels", no_plot)
    report = run(_pure(), "leiden", hypotheses=_hypotheses(), show=True, subcluster=False)
    printed = capsys.readouterr().out
    assert "Doublets" in printed
    assert "not recorded" in printed
    assert report["doublets"]["enabled"] is None
    assert printed.endswith(report["annotation_code"] + "\n")
    assert "cluster2celltype" in printed
    assert "fine_order" in printed


def test_gene_votes_share_one_wilcoxon_table(monkeypatch):
    adata = _pure()
    import scanpy as sc

    calls = {"n": 0}
    real = sc.tl.rank_genes_groups

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        assert kwargs.get("pts") is True
        return real(*args, **kwargs)

    monkeypatch.setattr(sc.tl, "rank_genes_groups", wrapped)
    from annotuneagent.workflow import _logfc_by_cluster

    tables = _logfc_by_cluster(adata, "leiden")
    wilcoxon_order(adata, "leiden")
    assert calls["n"] == 1
    assert tables["0"]["CD4"] > 0
    assert tables["0"]["CD8A"] <= 0
    assert "pts" in adata.uns["rank_genes_groups"]


def test_same_written_name_merges_clusters():
    x = np.zeros((60, 4), dtype=float)
    x[:20, 0] = 5
    x[:20, 1] = 5
    x[20:40, 0] = 5
    x[20:40, 2] = 5
    x[40:, 0] = 5
    x[40:, 1] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20 + ["2"] * 20})
    adata.var_names = ["CD3E", "CD4", "CD8A", "IL7R"]
    adata.uns["log1p"] = {"base": None}
    report = run(adata, "leiden", hypotheses=_hypotheses(), subcluster=False)
    assert report["merged"] == [{"name": "CD4 T", "cluster": "0", "members": ["0", "2"]}]
    same = adata.obs["leiden"].isin(["0", "2"])
    assert set(adata.obs.loc[same, "annotune_label"]) == {"CD4 T"}
    assert "annotune_cluster" not in adata.obs.columns


def test_custom_hypotheses_do_not_claim_literature_markers(tmp_path, monkeypatch):
    adata = _pure()
    path = tmp_path / "markers.png"
    report = run(adata, "leiden", hypotheses=_hypotheses(), dotplot_path=path, subcluster=False)
    assert report["all_labeled"] is True
    assert report["written"] is True
    assert set(adata.obs["annotune_label"]) == {"CD4 T", "CD8 T"}
    assert not path.exists()
    assert report["dotplot"] is None
    assert report["marker_dict"] == {}
    assert report["records"][0]["ranking"][0]["name"] in {"CD4 T", "CD8 T"}
    code = report["annotation_code"]
    assert "cluster2celltype" in code
    assert "'0': 'CD4 T'" in code or "'1': 'CD8 T'" in code
    assert "marker_dict" in code
    assert "fine_order" in code
    assert "sc.pl.dotplot" not in code
    monkeypatch.chdir(tmp_path)
    copied = adata.copy()
    exec(code, {"adata": copied})
    assert list(copied.obs["celltype"].astype(str)) == list(adata.obs["annotune_label"].astype(str))


def test_cited_enriched_markers_make_diagonal_dotplot(tmp_path, monkeypatch):
    genes = ["CD3E", "CD4", "IL7R", "MS4A1", "CD79A", "CD79B", "CD19", "BANK1"]
    x = np.zeros((40, len(genes)), dtype=float)
    x[:, genes.index("CD3E")] = 2  # Shared expression is not a diagonal marker.
    x[:20, genes.index("CD4")] = 5
    x[:20, genes.index("IL7R")] = 5
    for gene in ("MS4A1", "CD79A", "CD79B", "CD19", "BANK1"):
        x[20:, genes.index(gene)] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20})
    adata.var_names = genes
    adata.uns["log1p"] = {"base": None}
    path = tmp_path / "markers.pdf"
    names = ["CD4-positive, alpha-beta T cell", "B cell"]
    report = run(
        adata,
        "leiden",
        species="human",
        hypotheses=hypotheses_from_names(names, "human"),
        dotplot_path=path,
        subcluster=False,
    )
    assert report["dotplot"] == str(path)
    assert path.is_file()
    assert path.stat().st_size > 1000
    assert set(report["marker_dict"][names[0]]) == {"CD4", "IL7R"}
    assert set(report["marker_dict"][names[1]]) <= {"MS4A1", "CD79A", "CD79B", "CD19", "BANK1"}
    check = next(row for row in report["marker_validation"][names[0]] if row["gene"] == "CD3E")
    assert check["selected"] is False
    assert "not_specific_vs_other_labels" in check["reason"]
    assert "lit=PMID:" in check["literature"]
    code = report["annotation_code"]
    assert "marker_dict" in code and "fine_order" in code and "sc.pl.dotplot" in code
    monkeypatch.chdir(tmp_path)
    copied = adata.copy()
    exec(code, {"adata": copied})
    assert list(copied.obs["celltype"].astype(str)) == list(adata.obs["annotune_label"].astype(str))


def test_shared_enriched_marker_is_rejected_when_nothing_is_specific():
    from annotuneagent.marker_plot import select_marker_panels

    genes = ["NKG7", "GNLY", "KLRD1", "KLRF1", "CD3D", "CD3E", "CD3G", "TRAC"]
    x = np.zeros((90, len(genes)), dtype=float)
    for gene in ("NKG7", "GNLY", "KLRD1", "KLRF1"):
        x[:40, genes.index(gene)] = 1
        x[40:50, genes.index(gene)] = 1.5
    for gene in ("CD3D", "CD3E", "CD3G", "TRAC"):
        x[40:50, genes.index(gene)] = 5
    labels = ["NK cell"] * 40 + ["T cell"] * 10 + ["B cell"] * 40
    adata = AnnData(X=x, obs={"celltype": labels})
    adata.var_names = genes
    adata.uns["log1p"] = {"base": None}
    panels, checks, status = select_marker_panels(adata, "celltype", "human", None)
    assert status["NK cell"] == "no_marker_passed_expression_check"
    assert set(panels["NK cell"]) <= {"NKG7", "GNLY", "KLRD1", "KLRF1"}
    assert panels["NK cell"]
    shared = next(row for row in checks["NK cell"] if row["gene"] == "NKG7")
    assert shared["selected"] is False
    assert shared["reason"] == "not_specific_vs_other_labels"


def test_undetected_nk_call_keeps_the_marker_and_its_genes():
    genes = ["Avil", "Prph", "Nkg7", "Klrd1", "Klrb1c", "Ncr1", "Gzma", "Mpz", "Pmp22"]
    x = np.zeros((40, len(genes)), dtype=float)
    x[:20, genes.index("Avil")] = 4
    x[:20, genes.index("Prph")] = 4
    x[0, genes.index("Nkg7")] = 8
    x[20:, genes.index("Mpz")] = 4
    x[20:, genes.index("Pmp22")] = 4
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20})
    adata.var_names = genes
    adata.uns["log1p"] = {"base": None}

    def _nk(payload):
        return "Avil" in str(payload.get("genes"))

    def name(payload):
        if not _nk(payload):
            return Decision("escalate")
        assert "NK cell" not in {item["name"] for item in payload["dictionary_hits"]}
        return Decision("retest", ("NK cell",))

    def ml(payload):
        if not _nk(payload):
            return MlResult("escalate")
        return MlResult("named", ("NK cell",))

    names = ["sensory neuron", "myelinating Schwann cell", "NK cell"]
    report = run(
        adata,
        "leiden",
        species="mouse",
        tissue="drg",
        hypotheses=hypotheses_from_names(names, "mouse", "drg"),
        models=_chain(name, ml),
        subcluster=False,
    )
    assert list(adata.obs["annotune_label"].iloc[:20].unique()) == ["sensory neuron"]
    assert "NK cell" not in report["marker_dict"]
    assert set(report["marker_dict"]["sensory neuron"]) <= {"Avil", "Prph"}
    assert report["marker_dict"]["sensory neuron"]


def test_mixed_cluster_is_not_written_and_can_split(tmp_path, monkeypatch):
    x = np.zeros((80, 4), dtype=float)
    x[:40, 3] = 5
    x[40:60, 0] = 5
    x[40:60, 1] = 5
    x[60:, 0] = 5
    x[60:, 2] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 40 + ["1"] * 40})
    adata.var_names = ["CD3E", "CD4", "CD8A", "IL7R"]
    adata.uns["log1p"] = {"base": None}
    report = run(adata, "leiden", hypotheses=_hypotheses(), min_subcluster_cells=40)
    by_cluster = {record["cluster"]: record for record in report["records"]}
    assert by_cluster["1"]["status"] == "uncertain"
    assert report["written"] is True
    assert set(adata.obs["annotune_label"].iloc[40:]) == {"CD4 T", "CD8 T"}
    assert adata.obs["annotune_label"].iloc[:40].isna().all()
    assert report["queue"] == ["0"]
    assert "cluster_choices" in report["annotation_code"]
    assert "'0':" in report["annotation_code"]
    split = next(item for item in report["subclusters"] if item["parent"] == "1")
    child_status = {row["status"] for row in split["records"]}
    assert "label" in child_status
    child_rows = [row for row in report["clusters"] if str(row["cluster"]).startswith("1.")]
    assert len(child_rows) == 2
    for row in child_rows:
        assert row["query"]
        assert row["query"].splitlines()[1].startswith("Judgment:")
        assert "Enriched versus other cells:" in row["query"]
    assert "annotune_sub" in adata.obs.columns
    code = report["annotation_code"]
    for unit in adata.obs["annotune_sub"].iloc[40:].unique():
        assert repr(str(unit)) in code
    assert "adata.obs['annotune_sub']" in code
    monkeypatch.chdir(tmp_path)
    copied = adata.copy()
    exec(code, {"adata": copied, "pd": __import__("pandas")})
    got = copied.obs["celltype"].astype("string").fillna("")
    expected = adata.obs["annotune_label"].astype("string").fillna("")
    assert got.tolist() == expected.tolist()


def test_model_escalate_keeps_an_accepted_split():
    x = np.zeros((80, 4), dtype=float)
    x[:40, 3] = 5
    x[40:60, 0] = 5
    x[40:60, 1] = 5
    x[60:, 0] = 5
    x[60:, 2] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 40 + ["1"] * 40})
    adata.var_names = ["CD3E", "CD4", "CD8A", "IL7R"]
    adata.uns["log1p"] = {"base": None}

    report = run(
        adata,
        "leiden",
        hypotheses=_hypotheses(),
        models=_chain(lambda _payload: Decision("escalate"), lambda _payload: MlResult("escalate")),
        model_clusters=["1"],
        min_subcluster_cells=40,
    )
    parent = next(row for row in report["records"] if row["cluster"] == "1")
    assert parent["final_status"] == "named"
    assert parent["final_reason"] == "subcluster_labels"
    assert parent["written_label"] is None
    assert set(adata.obs["annotune_label"].iloc[40:]) == {"CD4 T", "CD8 T"}
    assert adata.obs["annotune_label"].iloc[:40].isna().all()
    assert report["queue"] == ["0"]
    assert "annotune_final_status" not in adata.obs.columns


def test_failed_subcluster_marker_keeps_the_label_and_flags_it(monkeypatch):
    x = np.zeros((80, 4), dtype=float)
    x[:40, 3] = 5
    x[40:60, :2] = 5
    x[60:, 0] = 5
    x[60:, 2] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 40 + ["1"] * 40})
    adata.var_names = ["CD3E", "CD4", "CD8A", "IL7R"]
    adata.uns["log1p"] = {"base": None}

    def marker_check(*_args):
        return (
            {},
            {},
            {
                "CD4 T": "no_marker_passed_expression_check",
                "CD8 T": "selected",
            },
        )

    monkeypatch.setattr("annotuneagent.workflow.select_marker_panels", marker_check)
    report = run(adata, "leiden", hypotheses=_hypotheses(), min_subcluster_cells=40)
    parent = next(row for row in report["records"] if row["cluster"] == "1")
    assert parent["final_status"] == "named"
    assert "no marker that passed" in parent["hint"]
    assert report["queue"] == ["0"]
    assert set(adata.obs["annotune_label"].iloc[40:60]) == {"CD4 T"}
    assert set(adata.obs["annotune_label"].iloc[60:]) == {"CD8 T"}
    child = next(row for row in report["subclusters"][0]["records"] if row["cell_type"] == "CD4 T")
    assert child["final_reason"] == "no_marker_passed"
    assert child["hint"] == "no marker passed"


def test_label_without_a_passing_marker_is_written_and_flagged():
    genes = ["CD3D", "CD3E", "CD3G", "TRAC", "MS4A1", "CD79A"]
    x = np.zeros((6, len(genes)), dtype=float)
    x[:3, :4] = 5
    x[3:, 4:] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 3 + ["1"] * 3})
    adata.var_names = genes
    adata.uns["log1p"] = {"base": None}
    report = run(
        adata,
        "leiden",
        species="human",
        hypotheses=hypotheses_from_names(["T cell"], "human", None),
        subcluster=False,
    )
    by_cluster = {row["cluster"]: row for row in report["records"]}
    assert by_cluster["0"]["final_status"] == "named"
    assert by_cluster["0"]["final_reason"] == "no_marker_passed"
    assert by_cluster["0"]["written_label"] == "T cell"
    assert "no marker passed" in by_cluster["0"]["hint"]
    assert set(adata.obs["annotune_label"].dropna()) == {"T cell"}
    assert "annotune_hint" not in adata.obs.columns
    assert report["queue"] == ["1"]


def test_failed_marker_name_yields_to_models_that_agree():
    from annotuneagent.workflow import _flag_labels_without_markers, _lineage_resources

    dag, label_map, _known = _lineage_resources("mouse")
    record = {
        "cluster": "7",
        "status": "label",
        "cell_type": "memory CD4 T cell",
        "written_label": "memory CD4 T cell",
        "final_status": "named",
        "final_reason": "evidence_conflict",
        "hint": "conflict: name classical monocyte, ML monocyte",
        "rounds": [
            {"step": "name", "names": ["classical monocyte"]},
            {"step": "ml", "names": ["monocyte"]},
        ],
    }
    resolved = {"7": {"7": "memory CD4 T cell"}}
    status = {"memory CD4 T cell": "no_marker_passed_expression_check"}
    changed = _flag_labels_without_markers(
        [record],
        resolved,
        status,
        {},
        species="mouse",
        dag=dag,
        label_map=label_map,
    )
    assert changed
    assert record["written_label"] == "classical monocyte"
    assert resolved["7"] == {"7": "classical monocyte"}
    assert record["hint"] == "conflict: marker memory CD4 T cell"


def test_failed_marker_call_does_not_replace_an_agreed_name():
    from annotuneagent.workflow import _flag_labels_without_markers, _lineage_resources

    dag, label_map, _known = _lineage_resources("mouse")
    record = {
        "cluster": "7",
        "status": "label",
        "cell_type": "memory CD4 T cell",
        "written_label": "classical monocyte",
        "final_status": "named",
        "final_reason": "evidence_conflict",
        "hint": "conflict: marker memory CD4 T cell",
        "rounds": [
            {"step": "name", "names": ["classical monocyte"]},
            {"step": "ml", "names": ["monocyte"]},
        ],
    }
    resolved = {"7": {"7": "classical monocyte"}}
    status = {"classical monocyte": "no_marker_passed_expression_check"}
    changed = _flag_labels_without_markers(
        [record],
        resolved,
        status,
        {},
        species="mouse",
        dag=dag,
        label_map=label_map,
        rejected={"classical monocyte", "memory CD4 T cell"},
    )
    assert changed is False
    assert record["written_label"] == "classical monocyte"
    assert record["final_reason"] == "no_marker_passed"
    assert "no marker passed" in record["hint"]


def test_ineligible_marker_score_does_not_win():
    from annotuneagent.workflow import _marker_score

    ranking = [
        {"name": "monocyte", "score": 1.2, "eligible": True},
        {"name": "memory CD4 T cell", "score": 4.0, "eligible": False},
    ]
    assert _marker_score("memory CD4 T cell", ranking) is None
    assert _marker_score("monocyte", ranking) == 1.2
    settled = settle_cluster(
        marker="cDC2",
        name="memory CD4 T cell",
        ml=MlResult("named", ("monocyte",)),
        added=[],
        cells=139,
        species="mouse",
        ranking=ranking,
        margin=0.4,
        marker_labeled=False,
    )
    assert settled["written_label"] == "monocyte"
    blocked = settle_cluster(
        marker="memory CD4 T cell",
        name=None,
        ml=MlResult("failed"),
        added=[],
        cells=100,
        ranking=[{"name": "memory CD4 T cell", "score": 4.0, "eligible": False}],
        marker_labeled=False,
    )
    assert blocked["written_label"] is None


def test_agreed_fibroblast_is_not_replaced_by_smooth_muscle():
    from annotuneagent.workflow import _flag_labels_without_markers, _lineage_resources

    dag, label_map, _known = _lineage_resources("mouse")
    record = {
        "cluster": "8",
        "status": "label",
        "cell_type": "smooth muscle cell",
        "written_label": "fibroblast",
        "final_status": "named",
        "final_reason": "evidence_agreement",
        "hint": None,
        "rounds": [
            {"step": "name", "names": ["fibroblast"]},
            {"step": "ml", "names": ["fibroblast"]},
        ],
    }
    resolved = {"8": {"8": "fibroblast"}}
    status = {
        "fibroblast": "no_marker_passed_expression_check",
        "smooth muscle cell": "selected",
    }
    changed = _flag_labels_without_markers(
        [record],
        resolved,
        status,
        {},
        species="mouse",
        dag=dag,
        label_map=label_map,
    )
    assert changed is False
    assert record["written_label"] == "fibroblast"
    assert resolved["8"] == {"8": "fibroblast"}
    assert record["final_reason"] == "no_marker_passed"
    assert record["hint"] == "no marker passed"


def test_agreed_name_yields_when_its_program_is_off():
    from annotuneagent.workflow import _flag_labels_without_markers, _lineage_resources

    dag, label_map, _known = _lineage_resources("mouse")
    record = {
        "cluster": "0",
        "status": "label",
        "cell_type": "sensory neuron",
        "written_label": "NK cell",
        "final_status": "named",
        "final_reason": "evidence_conflict",
        "hint": "conflict: marker sensory neuron",
        "rounds": [
            {"step": "name", "names": ["NK cell"]},
            {"step": "ml", "names": ["NK cell"]},
        ],
        "ranking": [
            {"name": "sensory neuron", "program_on": True, "program_specific": True},
            {"name": "NK cell", "program_on": False, "program_specific": False},
        ],
    }
    resolved = {"0": {"0": "NK cell"}}
    changed = _flag_labels_without_markers(
        [record],
        resolved,
        {"NK cell": "no_marker_passed_expression_check"},
        {},
        species="mouse",
        dag=dag,
        label_map=label_map,
    )
    assert changed
    assert record["written_label"] == "sensory neuron"
    assert resolved["0"] == {"0": "sensory neuron"}
    assert record["final_reason"] == "marker_rule"
    assert record["hint"] is None


def test_checked_marker_still_replaces_a_failed_name():
    from annotuneagent.workflow import _flag_labels_without_markers

    record = {
        "cluster": "0",
        "status": "label",
        "cell_type": "B cell",
        "written_label": "T cell",
        "final_status": "named",
        "final_reason": "evidence_conflict",
        "hint": "conflict: marker B cell",
        "rounds": [
            {"step": "name", "names": ["T cell"]},
            {"step": "ml", "names": ["NK cell"]},
        ],
    }
    resolved = {"0": {"0": "T cell"}}
    status = {
        "T cell": "no_marker_passed_expression_check",
        "B cell": "selected",
    }
    changed = _flag_labels_without_markers([record], resolved, status, {})
    assert changed
    assert record["written_label"] == "B cell"
    assert record["final_reason"] == "marker_rule"
    assert record["hint"] is None


def _chain(name, ml):
    return ModelChain(
        name=TaskEndpoint(ask=name),
        ml=TaskEndpoint(ask=ml),
    )


def test_model_retest_uses_training_payload_and_writes_the_name():
    adata = _pure()
    seen = []

    def name(payload):
        seen.append(payload)
        assert "marker" not in payload
        assert "dictionary_hits" in payload
        return Decision("retest", ("CD4 T",))

    def ml(payload):
        assert "candidates" in payload
        assert "marker" not in payload
        return MlResult("escalate")

    shared = [
        Hypothesis("CD8 T", positive=("CD3E", "CD8A"), negative=("CD4",)),
        Hypothesis("CD4 T", positive=("CD3E", "CD4"), negative=("CD8A",)),
    ]
    report = run(
        adata,
        "leiden",
        hypotheses=[shared[0]],
        catalog=shared,
        models=_chain(name, ml),
        model_clusters=["0"],
        subcluster=False,
    )
    assert seen and "dictionary_hits" in seen[0]
    cluster0 = next(row for row in report["records"] if row["cluster"] == "0")
    assert cluster0["written_label"] == "CD4 T"
    assert cluster0["final_reason"] == "check_rule"
    assert cluster0["hint"] is None
    assert set(adata.obs["annotune_label"].iloc[:20].astype(str)) == {"CD4 T"}


def test_name_and_ml_both_run_with_their_own_payloads():
    adata = _pure()
    calls = []

    def name(payload):
        calls.append(("name", "marker" in payload, "dictionary_hits" in payload))
        return Decision("retest", ("CD8 T",))

    def ml(payload):
        calls.append(("ml", "marker" in payload, "candidates" in payload))
        return MlResult("named", ("CD8 T",))

    shared = [
        Hypothesis("CD8 T", positive=("CD3E", "CD8A"), negative=("CD4",)),
        Hypothesis("CD4 T", positive=("CD3E", "CD4"), negative=("CD8A",)),
    ]
    report = run(
        adata,
        "leiden",
        hypotheses=[shared[0]],
        catalog=shared,
        models=_chain(name, ml),
        model_clusters=["0"],
        subcluster=False,
    )
    cluster0 = next(row for row in report["records"] if row["cluster"] == "0")
    assert [step["step"] for step in cluster0["rounds"]] == ["name", "check"]
    assert calls == [("name", False, True)]
    assert adata.obs["annotune_label"].iloc[0] != "CD4 T"


def test_unmapped_name_with_a_failed_marker_score_is_not_written():
    adata = _pure()

    def name(_payload):
        return Decision("retest", ("CD8 T",))

    def ml(_payload):
        return MlResult("escalate")

    report = run(
        adata,
        "leiden",
        hypotheses=[_hypotheses()[1]],
        catalog=_hypotheses(),
        models=_chain(name, ml),
        model_clusters=["0"],
        subcluster=False,
    )
    cluster0 = next(row for row in report["records"] if row["cluster"] == "0")
    assert cluster0["added"] == []
    assert cluster0["proposals"] == [{"source": "name", "name": "CD8 T"}]
    assert cluster0["written_label"] is None
    assert cluster0["final_status"] == "escalate"
    assert cluster0["final_reason"] == "check_rule"
    assert "What cell type is this cluster?" in cluster0["query"]
    assert "CD4" in cluster0["query"]
    assert "query" in next(row for row in report["clusters"] if row["cluster"] == "0")
    assert "0" in report["arbitration"]


def test_agreed_models_keep_a_name_when_no_panel_is_on():
    adata = _pure()

    def name(_payload):
        return Decision("retest", ("B cell",))

    def ml(_payload):
        return MlResult("named", ("B cell",))

    report = run(
        adata,
        "leiden",
        hypotheses=[_hypotheses()[1]],
        catalog=_hypotheses(),
        models=_chain(name, ml),
        model_clusters=["0"],
        subcluster=False,
    )
    cluster0 = next(row for row in report["records"] if row["cluster"] == "0")
    assert cluster0["written_label"] is None
    assert cluster0["final_reason"] == "check_rule"
    assert list(adata.obs["annotune_label"].iloc[:20].isna()) == [True] * 20


def test_name_ml_disagreement_writes_the_model_name_with_a_hint():
    adata = _pure()

    def name(_payload):
        return Decision("retest", ("CD4 T",))

    def ml(_payload):
        return MlResult("named", ("CD8 T",))

    report = run(
        adata,
        "leiden",
        hypotheses=[_hypotheses()[1]],
        catalog=_hypotheses(),
        models=_chain(name, ml),
        model_clusters=["0"],
        subcluster=False,
    )
    cluster0 = next(row for row in report["records"] if row["cluster"] == "0")
    assert cluster0["proposals"] == [{"source": "name", "name": "CD4 T"}]
    assert cluster0["written_label"] == "CD4 T"
    assert cluster0["final_status"] == "named"
    assert cluster0["final_reason"] == "check_rule"
    assert cluster0["hint"] is None
    assert cluster0["arbitration"] is False
    assert "0" not in report["arbitration"]
    assert set(adata.obs["annotune_label"].iloc[:20]) == {"CD4 T"}


def test_agreed_name_is_written_without_a_hint():
    settled = settle_cluster(
        marker="B cell",
        name="B cell",
        ml=MlResult("named", ("B cell",)),
        added=[],
        cells=100,
        ranking=[
            {
                "name": "B cell",
                "score": 1.0,
                "program_on": True,
                "program_specific": True,
                "n_positive": 2,
            }
        ],
    )
    assert settled["written_label"] == "B cell"
    assert settled["hint"] is None
    assert settled["final_reason"] == "evidence_agreement"
    assert settled["final_status"] == "named"


def test_one_gene_agreement_stays_written():
    settled = settle_cluster(
        marker="TA",
        name="TA",
        ml=MlResult("escalate"),
        added=[],
        cells=40,
        ranking=[
            {
                "name": "TA",
                "score": 1.0,
                "program_on": False,
                "program_specific": False,
                "n_positive": 1,
            }
        ],
    )
    assert settled["written_label"] == "TA"
    assert settled["hint"] is None
    assert settled["final_reason"] == "evidence_agreement"


def test_agreement_without_specific_markers_stays_written_for_review():
    detected = settle_cluster(
        marker="B cell",
        name="B cell",
        ml=MlResult("named", ("B cell",)),
        added=[],
        cells=100,
        ranking=[
            {
                "name": "B cell",
                "score": 1.0,
                "program_on": True,
                "program_specific": False,
                "n_positive": 2,
            }
        ],
    )
    assert detected["written_label"] == "B cell"
    assert detected["final_status"] == "review"
    assert detected["final_reason"] == "weak_agreement"
    assert detected["hint"] == "weak agreement on B cell"
    gap = settle_cluster(
        marker="innate lymphoid cell",
        name="innate lymphoid cell",
        ml=MlResult("named", ("innate lymphoid cell",)),
        added=[],
        cells=149,
        ranking=[
            {
                "name": "innate lymphoid cell",
                "score": 0.2,
                "program_on": False,
                "program_specific": False,
                "n_positive": 4,
            }
        ],
        unexplained=["KRT8", "KRT18"],
    )
    assert gap["written_label"] == "innate lymphoid cell"
    assert gap["final_status"] == "review"
    assert gap["final_reason"] == "model_agreement"
    assert gap["hint"] == "unexplained: KRT8, KRT18"


def test_model_agreement_keeps_the_finer_name_when_no_panel_is_on():
    settled = settle_cluster(
        marker="endothelial cell",
        name="endothelial cell",
        ml=MlResult("named", ("arterial endothelial cell",)),
        added=[],
        cells=200,
        species="human",
        ranking=[_absent("endothelial cell", 0.4), _absent("arterial endothelial cell", 0.2)],
        unexplained=["GJA5", "HEY1"],
    )
    assert settled["written_label"] == "arterial endothelial cell"
    assert settled["final_reason"] == "model_agreement"
    assert "GJA5" in settled["hint"]


def test_broad_smooth_muscle_yields_to_a_specific_marker():
    basal = "basal-myoepithelial cell of mammary gland"
    settled = settle_cluster(
        marker=basal,
        name="smooth muscle cell",
        ml=None,
        added=[],
        cells=7435,
        species="mouse",
        ranking=[
            {
                "name": basal,
                "score": 2.0,
                "program_on": True,
                "program_specific": True,
                "n_positive": 2,
            },
            {
                "name": "smooth muscle cell",
                "score": 1.0,
                "program_on": True,
                "program_specific": False,
                "n_positive": 5,
            },
        ],
        marker_labeled=True,
        check={"action": "agree", "name": "smooth muscle cell"},
    )
    assert settled["written_label"] == basal
    assert settled["final_reason"] == "check_rule"


def test_broad_fibroblast_yields_to_theca_and_not_to_smooth_muscle():
    theca = settle_cluster(
        marker="theca cell",
        name="fibroblast",
        ml=None,
        added=[],
        cells=3390,
        species="human",
        ranking=[
            {
                "name": "theca cell",
                "score": 1.4,
                "program_on": True,
                "program_specific": True,
                "n_positive": 2,
            },
            _absent("fibroblast", 0.4),
        ],
        marker_labeled=True,
        check={"action": "agree", "name": "fibroblast"},
    )
    assert theca["written_label"] == "theca cell"
    muscle = settle_cluster(
        marker="smooth muscle cell",
        name="fibroblast",
        ml=None,
        added=[],
        cells=115,
        species="mouse",
        ranking=[
            {
                "name": "smooth muscle cell",
                "score": 1.4,
                "program_on": True,
                "program_specific": True,
                "n_positive": 5,
            },
            {
                "name": "fibroblast",
                "score": 1.0,
                "program_on": True,
                "program_specific": False,
                "n_positive": 5,
            },
        ],
        marker_labeled=True,
        check={"action": "agree", "name": "fibroblast"},
    )
    assert muscle["written_label"] == "fibroblast"


def test_skeletal_myoblast_replaces_a_broad_smooth_muscle_title():
    settled = settle_cluster(
        marker="skeletal muscle myoblast",
        name="skeletal muscle myoblast",
        ml=None,
        added=[],
        cells=2060,
        species="mouse",
        ranking=[
            {
                "name": "skeletal muscle myoblast",
                "score": 1.5,
                "program_on": True,
                "program_specific": True,
                "n_positive": 2,
            },
            {
                "name": "smooth muscle cell",
                "score": 1.0,
                "program_on": True,
                "program_specific": False,
                "n_positive": 5,
            },
        ],
        marker_labeled=True,
        check={"action": "replace", "name": "smooth muscle cell"},
    )
    assert settled["written_label"] == "skeletal muscle myoblast"


def test_check_agreement_keeps_plasma_cell():
    settled = settle_cluster(
        marker="plasmablast",
        name="plasma cell",
        ml=None,
        added=[],
        cells=110,
        species="human",
        ranking=[
            {
                "name": "plasmablast",
                "score": 1.2,
                "program_on": True,
                "program_specific": True,
                "n_positive": 3,
            },
            {
                "name": "plasma cell",
                "score": 1.0,
                "program_on": True,
                "program_specific": True,
                "n_positive": 3,
            },
        ],
        marker_labeled=True,
        check={"action": "agree", "name": "plasma cell"},
    )
    assert settled["written_label"] == "plasma cell"
    assert settled["final_reason"] == "check_rule"
    assert settled["hint"] is None


def test_check_agreement_keeps_a_name_the_dictionary_does_not_detect():
    settled = settle_cluster(
        marker=None,
        name="basal cell of prostate epithelium",
        ml=None,
        added=[],
        cells=3740,
        species="human",
        ranking=[_absent("basal cell of prostate epithelium", 0.2)],
        unexplained=["KRT15", "HSPA1B"],
        check={"action": "agree", "name": "basal cell of prostate epithelium"},
    )
    assert settled["written_label"] == "basal cell of prostate epithelium"
    assert settled["final_status"] == "named"
    assert settled["final_reason"] == "check_rule"
    assert settled["hint"] == "unexplained: KRT15, HSPA1B"


def test_a_name_check_did_not_accept_is_still_outside_the_dictionary():
    settled = settle_cluster(
        marker=None,
        name="mesothelial cell",
        ml=None,
        added=[],
        cells=469,
        species="human",
        ranking=[_absent("mesothelial cell", 0.4)],
        unexplained=["AZGP1"],
        check={"action": "escalate", "name": None},
    )
    assert settled["written_label"] is None
    assert settled["final_reason"] == "check_rule"


def test_name_and_marker_agreement_survives_a_dictionary_gap():
    settled = settle_cluster(
        marker="basal cell of prostate epithelium",
        name="basal cell of prostate epithelium",
        ml=None,
        added=[],
        cells=4841,
        species="human",
        ranking=[_absent("basal cell of prostate epithelium", 0.2)],
        unexplained=["KRT15", "DST"],
        check={"action": "escalate", "name": None},
    )
    assert settled["written_label"] == "basal cell of prostate epithelium"
    assert settled["final_reason"] == "model_agreement"
    assert settled["hint"] == "unexplained: KRT15, DST"


def test_broad_parent_yields_to_a_specific_descendant():
    settled = settle_cluster(
        marker="alveolar fibroblast",
        name="fibroblast",
        ml=None,
        added=[],
        cells=1493,
        species="mouse",
        ranking=[
            {
                "name": "alveolar fibroblast",
                "score": 1.4,
                "program_on": True,
                "program_specific": True,
                "n_positive": 4,
            },
            {
                "name": "fibroblast",
                "score": 1.0,
                "program_on": True,
                "program_specific": False,
                "n_positive": 5,
            },
        ],
        marker_labeled=True,
        check={"action": "agree", "name": "fibroblast"},
    )
    assert settled["written_label"] == "alveolar fibroblast"


def test_check_agreement_yields_when_its_program_is_off():
    settled = settle_cluster(
        marker="non-classical monocyte",
        name="alveolar macrophage",
        ml=None,
        added=[],
        cells=1025,
        species="mouse",
        ranking=[
            {
                "name": "non-classical monocyte",
                "score": 1.2,
                "program_on": True,
                "program_specific": True,
                "n_positive": 4,
            },
            _absent("alveolar macrophage", 0.2),
        ],
        marker_labeled=True,
        check={"action": "agree", "name": "alveolar macrophage"},
    )
    assert settled["written_label"] == "non-classical monocyte"


def test_disagreeing_models_still_drop_a_name_outside_the_dictionary():
    settled = settle_cluster(
        marker="B cell",
        name="B cell",
        ml=MlResult("named", ("T cell",)),
        added=[],
        cells=100,
        species="human",
        ranking=[_absent("B cell", 0.4), _absent("T cell", 0.2)],
        unexplained=["MS4A1"],
    )
    assert settled["written_label"] is None
    assert settled["final_reason"] == "out_of_dictionary"
    assert settled["hint"] == "unexplained: MS4A1"


def _absent(name: str, score: float) -> dict:
    return {
        "name": name,
        "score": score,
        "program_on": False,
        "program_specific": False,
        "n_positive": 5,
    }


def test_endothelial_is_written_when_pericyte_genes_are_absent():
    settled = settle_cluster(
        marker="pericyte",
        name="endothelial cell",
        ml=MlResult("named", ("pericyte",)),
        added=[],
        cells=342,
        ranking=[_absent("pericyte", 0.9), _absent("endothelial cell", 0.2)],
        unexplained=["Emcn", "Gng11", "Col4a1"],
    )
    assert settled["written_label"] == "endothelial cell"
    assert settled["final_status"] == "review"
    assert settled["final_reason"] == "endothelial_not_pericyte"
    assert settled["hint"] == "endothelial cell rather than pericyte"
    blocked = settle_cluster(
        marker="pericyte",
        name="endothelial cell",
        ml=MlResult("named", ("pericyte",)),
        added=[],
        cells=342,
        ranking=[_absent("pericyte", 0.9), _absent("endothelial cell", 0.2)],
        unexplained=["Emcn", "Rgs5"],
    )
    assert blocked["written_label"] is None
    assert blocked["final_reason"] == "out_of_dictionary"
    plain = settle_cluster(
        marker="pericyte",
        name="endothelial cell",
        ml=MlResult("named", ("pericyte",)),
        added=[],
        cells=342,
        ranking=[_absent("pericyte", 0.9), _absent("endothelial cell", 0.2)],
        unexplained=["Col4a1", "Igfbp7"],
    )
    assert plain["written_label"] is None


def _force_half_split(part, **_kwargs):
    n = part.n_obs
    part.obs[wf.SUBCLUSTER_KEY] = np.array(["0"] * (n // 2) + ["1"] * (n - n // 2))
    return True


def test_a_uniform_split_is_collapsed(monkeypatch):
    monkeypatch.setattr(wf, "_subcluster", _force_half_split)
    genes = ["CD3E", "IL7R", "CD8A", "CD8B", "MS4A1"]
    x = np.zeros((120, len(genes)), dtype=float)
    x[:80, :4] = 5
    x[80:, 4] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 80 + ["1"] * 40})
    adata.var_names = genes
    adata.uns["log1p"] = {"base": None}
    cd4 = "CD4-positive, alpha-beta T cell"
    report = run(
        adata,
        "leiden",
        species="human",
        hypotheses=hypotheses_from_names(
            [cd4, "CD8-positive, alpha-beta T cell", "B cell"], "human"
        ),
        models=_chain(
            lambda _payload: Decision("retest", (cd4,)),
            lambda _payload: MlResult("named", (cd4,)),
        ),
        model_clusters=["0"],
        min_subcluster_cells=80,
        min_margin=2,
    )
    assert "annotune_sub" not in adata.obs.columns
    assert report["subclusters"] == []
    assert set(adata.obs.loc[adata.obs["leiden"] == "0", "annotune_label"].astype(str)) == {
        "T cell"
    }
    assert "0.0" not in report["annotation_code"]
    assert "'0': 'T cell'" in report["annotation_code"]
    choices = report["annotation_code"].split("# unresolved clusters")
    assert len(choices) == 1 or "'0':" not in choices[1]


def test_install_split_extends_a_categorical_subcluster_column():
    adata = AnnData(X=np.zeros((6, 1)), obs={"leiden": pd.Categorical(["0"] * 3 + ["1"] * 3)})
    adata.obs[wf.SUBCLUSTER_KEY] = pd.Categorical(adata.obs["leiden"].astype(str))
    part = adata[:3].copy()
    part.obs[wf.SUBCLUSTER_KEY] = ["0", "1", "0"]
    wf._install_split(adata, "leiden", np.array([True, True, True, False, False, False]), part, "0")
    assert list(adata.obs[wf.SUBCLUSTER_KEY].astype(str)) == ["0.0", "0.1", "0.0", "1", "1", "1"]


def test_a_named_cluster_is_split_when_name_and_ml_differ(monkeypatch):
    def force(part, **_kwargs):
        n = part.n_obs
        part.obs[wf.SUBCLUSTER_KEY] = np.array(["0"] * 70 + ["1"] * (n - 70))
        return True

    monkeypatch.setattr(wf, "_subcluster", force)
    genes = ["CD3E", "IL7R", "CD8A", "CD8B", "MS4A1"]
    x = np.zeros((120, len(genes)), dtype=float)
    x[:70, 0] = 5
    x[:70, 1] = 5
    x[70:80, 0] = 5
    x[70:80, 2] = 5
    x[70:80, 3] = 5
    x[80:, 4] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 80 + ["1"] * 40})
    adata.var_names = genes
    adata.uns["log1p"] = {"base": None}
    cd4 = "CD4-positive, alpha-beta T cell"
    cd8 = "CD8-positive, alpha-beta T cell"
    report = run(
        adata,
        "leiden",
        species="human",
        hypotheses=hypotheses_from_names([cd4, cd8, "B cell"], "human"),
        models=_chain(
            lambda _payload: Decision("retest", (cd8,)),
            lambda _payload: MlResult("named", (cd4,)),
        ),
        model_clusters=["0"],
        min_subcluster_cells=80,
    )
    labels = adata.obs.loc[adata.obs["leiden"].astype(str) == "0", "annotune_label"].astype(str)
    assert set(labels.unique()) == {cd4, cd8}
    parent = next(row for row in report["records"] if row["cluster"] == "0")
    assert parent["written_label"] is None
    assert parent["final_reason"] == "subcluster_labels"
    assert any(item["parent"] == "0" for item in report["subclusters"])
    assert "'0.0'" in report["annotation_code"]
    assert "'0.1'" in report["annotation_code"]


def test_endothelial_pericyte_split_leaves_double_positive_cells_unlabeled(monkeypatch):
    def force(part, **_kwargs):
        part.obs[wf.SUBCLUSTER_KEY] = np.array(["0"] * 55 + ["1"] * (part.n_obs - 55))
        return True

    def fake_annotate(
        adata, cluster_key, hypotheses, min_margin, ranker, conformal_q, clusters=None
    ):
        if cluster_key == wf._SPLIT_SCORE_KEY:
            names = {"split:0": "endothelial cell", "split:1": "pericyte"}
            ids = list(clusters) if clusters is not None else list(names)
            return [_child(str(item), names[str(item)]) for item in ids]
        if cluster_key == wf.SUBCLUSTER_KEY:
            return [
                _child("0", "endothelial cell"),
                _child("1", "pericyte"),
            ]
        rows = []
        for cluster in adata.obs[cluster_key].astype(str).unique():
            if clusters is not None and str(cluster) not in {str(item) for item in clusters}:
                continue
            name = "endothelial cell" if str(cluster) == "0" else "B cell"
            rows.append(_child(str(cluster), name))
        return rows

    monkeypatch.setattr(wf, "_subcluster", force)
    monkeypatch.setattr(wf, "_annotate", fake_annotate)
    genes = ["Cdh5", "Cldn5", "Kdr", "Rgs5", "Pdgfrb"]
    x = np.zeros((110, len(genes)), dtype=float)
    x[:55, 0:3] = 5
    x[40:55, 3:5] = 5
    x[55:90, 3:5] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 90 + ["1"] * 20})
    adata.var_names = genes
    adata.uns["log1p"] = {"base": None}
    report = run(
        adata,
        "leiden",
        species="mouse",
        hypotheses=hypotheses_from_names(["endothelial cell", "pericyte", "B cell"], "mouse"),
        models=_chain(
            lambda _payload: Decision("retest", ("endothelial cell",)),
            lambda _payload: MlResult("named", ("pericyte",)),
        ),
        model_clusters=["0"],
        min_subcluster_cells=40,
    )
    labels = adata.obs["annotune_label"]
    assert set(labels.iloc[:90].dropna().astype(str)) == {"endothelial cell"}
    assert int(labels.iloc[:90].isna().sum()) == 0
    parent = next(row for row in report["records"] if row["cluster"] == "0")
    assert parent["written_label"] == "endothelial cell"
    assert parent["final_reason"] == "check_rule"
    assert "doublet" not in report["annotation_code"]


def _program(row: dict, name: str) -> bool:
    return any(
        item.get("name") == name and item.get("program_specific") is True
        for item in row.get("ranking") or []
    )


def test_one_specific_mural_program_keeps_its_name():
    row = _child("1", "smooth muscle cell")
    row["ranking"].append(
        {"name": "pericyte", "score": 0.2, "program_on": True, "program_specific": False}
    )
    assert wf._withhold_shared_mural(row) is False
    assert row["cell_type"] == "smooth muscle cell"


def test_mural_child_stays_unlabeled_when_both_programs_are_specific(monkeypatch):
    def force(part, **_kwargs):
        part.obs[wf.SUBCLUSTER_KEY] = np.array(["0"] * 40 + ["1"] * (part.n_obs - 40))
        return True

    monkeypatch.setattr(wf, "_subcluster", force)
    genes = [
        "Pecam1",
        "Cdh5",
        "Kdr",
        "Cldn5",
        "Flt1",
        "Rgs5",
        "Pdgfrb",
        "Kcnj8",
        "Abcc9",
        "Higd1b",
        "Acta2",
        "Myh11",
        "Cnn1",
        "Des",
        "Tagln",
    ]
    x = np.zeros((320, len(genes)), dtype=float)
    x[:40, 5:10] = 2
    x[:40, [10, 11, 13, 14]] = 2
    x[40:80, 0:5] = 2
    # Eleven endothelial cells also express three pericyte genes. Inside the
    # parent that makes those genes look nonspecific. Against the other cells
    # they stay specific.
    x[69:80, 5:8] = 2
    x[80:160, 10] = 2
    adata = AnnData(X=x, obs={"leiden": ["0"] * 80 + ["1"] * 240})
    adata.var_names = genes
    adata.uns["log1p"] = {"base": None}
    hyps = hypotheses_from_names(["endothelial cell", "pericyte", "smooth muscle cell"], "mouse")
    part = adata[:80].copy()
    part.obs[wf.SUBCLUSTER_KEY] = np.array(["0"] * 40 + ["1"] * 40)
    sibling = {
        row["cluster"]: row for row in wf._annotate(part, wf.SUBCLUSTER_KEY, hyps, 0.15, None, None)
    }
    assert _program(sibling["0"], "pericyte") is False
    assert _program(sibling["0"], "smooth muscle cell") is True
    report = run(
        adata,
        "leiden",
        species="mouse",
        hypotheses=hyps,
        models=_chain(
            lambda _payload: Decision("retest", ("endothelial cell",)),
            lambda _payload: MlResult("named", ("pericyte",)),
        ),
        model_clusters=["0"],
        min_subcluster_cells=40,
    )
    labels = adata.obs["annotune_label"]
    assert set(labels.iloc[:80].dropna().astype(str)) == {"endothelial cell"}
    assert int(labels.iloc[:80].isna().sum()) == 0
    parent = next(row for row in report["records"] if row["cluster"] == "0")
    assert parent["written_label"] == "endothelial cell"
    assert parent["final_reason"] == "check_rule"
    assert "0.0" not in {row["cluster"] for row in report["clusters"]}


def _child(cluster: str, name: str | None) -> dict:
    labeled = name is not None
    return {
        "cluster": cluster,
        "status": "label" if labeled else "uncertain",
        "cell_type": name,
        "margin": 0.5,
        "ranking": [
            {
                "name": name or "endothelial cell",
                "score": 1.0,
                "program_on": True,
                "program_specific": True,
                "eligible": True,
                "n_positive": 3,
            }
        ],
        "evidence": {"de_not_in_markers": []},
    }


def test_a_nearest_name_without_majority_markers_is_not_written():
    x = np.zeros((40, 4), dtype=float)
    x[:20, 0] = 5
    x[20:, 2] = 5
    x[20:, 3] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20})
    adata.var_names = ["A", "B", "C", "D"]
    adata.uns["log1p"] = {"base": None}
    report = run(
        adata,
        "leiden",
        hypotheses=[
            Hypothesis("Blood cell", positive=("A", "B")),
            Hypothesis("Other cell", positive=("C", "D")),
        ],
        subcluster=False,
    )
    by_id = {row["cluster"]: row for row in report["records"]}
    assert by_id["0"]["final_status"] == "out_of_dictionary"
    assert by_id["0"]["written_label"] is None
    assert by_id["0"]["hint"]
    assert by_id["0"]["query"]
    assert by_id["1"]["written_label"] == "Other cell"
    assert by_id["1"]["final_reason"] == "marker_rule"
    assert "What cell type is this cluster?" in by_id["1"]["query"]
    assert by_id["1"]["ranking"][0]["presence"] > 0.5
    assert list(adata.obs["annotune_label"].iloc[:20].isna()) == [True] * 20
    assert set(adata.obs["annotune_label"].iloc[20:].astype(str)) == {"Other cell"}
    assert any(row["cluster"] == "0" for row in report["review"])


def test_finer_lineage_name_is_written_when_its_marker_score_passes():
    ranking = [
        {
            "name": "CD14-positive monocyte",
            "score": 0.8,
            "program_on": True,
            "program_specific": True,
            "n_positive": 2,
        },
        {"name": "monocyte", "score": 0.3, "program_on": True, "program_specific": True},
    ]
    settled = settle_cluster(
        marker="CD14-positive monocyte",
        name="monocyte",
        ml=MlResult("named", ("monocyte",)),
        added=[],
        cells=100,
        species="human",
        ranking=ranking,
        margin=0.5,
        marker_labeled=True,
    )
    assert settled["written_label"] == "CD14-positive monocyte"
    assert settled["hint"] is None
    assert settled["final_reason"] == "evidence_agreement"


def test_coarser_name_is_kept_when_the_finer_marker_score_fails():
    ranking = [
        {"name": "monocyte", "score": 0.8, "program_on": True, "program_specific": True},
        {"name": "CD14-positive monocyte", "score": 0.2, "program_on": False},
    ]
    settled = settle_cluster(
        marker="monocyte",
        name="CD14-positive monocyte",
        ml=MlResult("named", ("monocyte",)),
        added=[],
        cells=100,
        species="human",
        ranking=ranking,
        margin=0.6,
        marker_labeled=True,
    )
    assert settled["written_label"] == "monocyte"
    assert settled["hint"] is None


def test_sibling_call_is_a_conflict():
    ranking = [
        {"name": "NK cell", "score": 0.7},
        {"name": "gamma-delta T cell", "score": 0.2},
    ]
    settled = settle_cluster(
        marker="NK cell",
        name="NK cell",
        ml=MlResult("named", ("gamma-delta T cell",)),
        added=[],
        cells=100,
        species="human",
        ranking=ranking,
        margin=0.5,
        marker_labeled=True,
    )
    assert settled["written_label"] == "NK cell"
    assert settled["hint"] == "weak agreement on NK cell"


def test_classical_and_cd14_monocytes_do_not_share_a_lineage():
    ranking = [
        {"name": "classical monocyte", "score": 0.6},
        {"name": "CD14-positive monocyte", "score": 0.4},
    ]
    settled = settle_cluster(
        marker="classical monocyte",
        name="CD14-positive monocyte",
        ml=MlResult("named", ("CD14-positive monocyte",)),
        added=[],
        cells=80,
        species="human",
        ranking=ranking,
        margin=0.2,
        marker_labeled=True,
    )
    assert settled["written_label"] == "CD14-positive monocyte"
    assert settled["hint"] == "conflict: marker classical monocyte"


def test_three_lineages_write_the_higher_marker_score_with_a_hint():
    ranking = [
        {"name": "monocyte", "score": 0.8},
        {"name": "NK cell", "score": 0.2},
        {"name": "B cell", "score": 0.1},
    ]
    settled = settle_cluster(
        marker="monocyte",
        name="NK cell",
        ml=MlResult("named", ("B cell",)),
        added=[],
        cells=50,
        species="human",
        ranking=ranking,
        margin=0.6,
        marker_labeled=True,
    )
    assert settled["written_label"] == "monocyte"
    assert settled["hint"] == "conflict: name NK cell"


def test_close_margin_writes_the_descendant_both_models_name():
    ranking = [
        {"name": "monocyte", "score": 0.659},
        {"name": "cDC2", "score": 0.517},
        {"name": "CD14-positive monocyte", "score": 0.419},
    ]
    settled = settle_cluster(
        marker="monocyte",
        name="CD14-positive monocyte",
        ml=MlResult("named", ("CD14-positive monocyte",)),
        added=[],
        cells=1271,
        species="human",
        ranking=ranking,
        margin=0.142,
        marker_labeled=False,
    )
    assert settled["written_label"] == "CD14-positive monocyte"
    assert settled["hint"] == "close marker scores"
    assert settled["final_reason"] == "evidence_conflict"


def test_platelet_aggregate_is_named_as_the_monocyte():
    ranking = [
        {"name": "platelet", "score": 1.73, "program_on": True},
        {"name": "megakaryocyte", "score": 1.34, "program_on": True},
        {"name": "CD14-positive monocyte", "score": 0.50, "program_on": True},
        {"name": "monocyte", "score": 0.37, "program_on": True},
    ]
    settled = settle_cluster(
        marker="platelet",
        name="platelet",
        ml=MlResult("named", ("megakaryocyte",)),
        added=[],
        cells=123,
        species="human",
        ranking=ranking,
        margin=0.39,
        marker_labeled=True,
    )
    assert settled["written_label"] == "CD14-positive monocyte"
    assert settled["hint"] == "conflict: marker platelet"


def test_platelet_without_a_monocyte_program_keeps_its_name():
    ranking = [
        {"name": "platelet", "score": 6.47, "program_on": True},
        {"name": "megakaryocyte", "score": 4.98, "program_on": True},
        {"name": "monocyte", "score": 0.1, "program_on": False},
    ]
    settled = settle_cluster(
        marker="platelet",
        name="platelet",
        ml=MlResult("named", ("megakaryocyte",)),
        added=[],
        cells=175,
        species="human",
        ranking=ranking,
        margin=1.5,
        marker_labeled=True,
    )
    assert settled["written_label"] == "platelet"
    assert settled["hint"] == "weak agreement on platelet"


def test_small_cluster_with_two_programs_keeps_a_unanimous_name():
    ranking = [
        {"name": "NK cell", "score": 1.70, "program_on": True, "program_specific": True},
        {"name": "CD56dim NK cell", "score": 1.47, "program_on": True},
        {"name": "monocyte", "score": 0.40, "program_on": True},
    ]
    mixed = settle_cluster(
        marker="NK cell",
        name="NK cell",
        ml=MlResult("named", ("NK cell",)),
        added=[],
        cells=22,
        species="human",
        ranking=ranking,
        margin=0.23,
        marker_labeled=True,
    )
    assert mixed["written_label"] == "NK cell"
    assert mixed["hint"] == "cell programs: NK cell, CD56dim NK cell, monocyte"
    assert mixed["final_status"] == "named"
    assert mixed["final_reason"] == "mixed_programs"
    disagreed = settle_cluster(
        marker="NK cell",
        name="monocyte",
        ml=MlResult("named", ("NK cell",)),
        added=[],
        cells=22,
        species="human",
        ranking=ranking,
        margin=0.23,
        marker_labeled=True,
    )
    assert disagreed["written_label"] is None
    assert disagreed["final_status"] == "review"
    assert disagreed["final_reason"] == "mixed_programs"
    kept = settle_cluster(
        marker="NK cell",
        name="NK cell",
        ml=MlResult("named", ("NK cell",)),
        added=[],
        cells=654,
        species="human",
        ranking=ranking,
        margin=0.57,
        marker_labeled=True,
    )
    assert kept["written_label"] == "NK cell"
    assert kept["hint"] is None


def test_program_rules_apply_when_the_marker_already_labeled():
    from annotuneagent.workflow import _lineage_resources, _program_hold

    dag, label_map, _names = _lineage_resources()
    platelet = {
        "status": "label",
        "cell_type": "platelet",
        "ranking": [
            {"name": "platelet", "score": 1.73, "program_on": True},
            {"name": "CD14-positive monocyte", "score": 0.50, "program_on": True},
            {"name": "monocyte", "score": 0.37, "program_on": True},
        ],
    }
    held = _program_hold(platelet, 123, "human", dag, label_map)
    assert held["written_label"] == "CD14-positive monocyte"
    assert held["hint"] == "conflict: marker platelet"
    pure = _program_hold(
        {
            "status": "label",
            "cell_type": "platelet",
            "ranking": [
                {"name": "platelet", "score": 6.47, "program_on": True},
                {"name": "monocyte", "score": 0.1, "program_on": False},
            ],
        },
        175,
        "human",
        dag,
        label_map,
    )
    assert pure is None
    mixed = _program_hold(
        {
            "status": "label",
            "cell_type": "NK cell",
            "ranking": [
                {"name": "NK cell", "score": 1.70, "program_on": True},
                {"name": "monocyte", "score": 0.40, "program_on": True},
            ],
        },
        22,
        "human",
        dag,
        label_map,
    )
    assert mixed["written_label"] is None
    assert mixed["hint"] == "cell programs: NK cell, monocyte"
    large = {
        "status": "label",
        "cell_type": "NK cell",
        "ranking": [
            {"name": "NK cell", "score": 1.70, "program_on": True},
            {"name": "monocyte", "score": 0.40, "program_on": True},
        ],
    }
    assert _program_hold(large, 654, "human", dag, label_map) is None


def test_close_marker_scores_add_a_hint():
    ranking = [
        {"name": "monocyte", "score": 0.40},
        {"name": "B cell", "score": 0.30},
    ]
    settled = settle_cluster(
        marker="monocyte",
        name="monocyte",
        ml=MlResult("named", ("monocyte",)),
        added=[],
        cells=100,
        species="human",
        ranking=ranking,
        margin=0.10,
        marker_labeled=False,
    )
    assert settled["written_label"] == "monocyte"
    assert settled["hint"] == "close marker scores"
    assert settled["final_reason"] == "evidence_conflict"


def test_ontology_name_is_kept_when_the_model_name_is_outside_the_map():
    settled = settle_cluster(
        marker="B cell",
        name="author B",
        ml=MlResult("named", ("author B",)),
        added=[],
        cells=100,
        ontology_names={"B cell"},
    )
    assert settled["written_label"] == "B cell"
    assert settled["hint"] == "conflict: name author B"
    assert settled["final_status"] == "named"


def test_failed_call_keeps_a_marker_name_with_a_hint():
    settled = settle_cluster(
        marker="pancreatic beta cell",
        name=None,
        ml=MlResult("failed"),
        added=[],
        cells=100,
        failed=True,
        marker_labeled=True,
    )
    assert settled["written_label"] == "pancreatic beta cell"
    assert settled["hint"] == "model call failed"
    assert settled["final_status"] == "named"
    assert settled["final_reason"] == "model_failure"


def test_final_status_for_small_empty_and_failed_clusters():
    common = {"marker": None, "name": None, "added": []}
    small = settle_cluster(**common, ml=MlResult("named", ("CD4 T",)), cells=10)
    assert (
        small["final_status"],
        small["final_reason"],
        small["written_label"],
        small["hint"],
    ) == (
        "named",
        "small_cluster",
        "CD4 T",
        "fewer than 20 cells",
    )
    bare = settle_cluster(**common, ml=MlResult("escalate"), cells=10)
    assert (bare["final_status"], bare["final_reason"], bare["written_label"]) == (
        "review",
        "small_cluster",
        None,
    )
    empty = settle_cluster(**common, ml=MlResult("escalate"), cells=40)
    assert (empty["final_status"], empty["final_reason"]) == ("escalate", "no_supported_name")
    failed = settle_cluster(**common, ml=MlResult("failed"), cells=40)
    assert (failed["final_status"], failed["final_reason"]) == ("review", "model_failure")


def test_absent_negative_gene_keeps_the_type():
    from annotuneagent.workflow import _in_matrix

    hypotheses = [
        Hypothesis("cDC2", positive=("CD1C", "FCER1A"), negative=("CLEC9A", "CD14")),
        Hypothesis("megakaryocyte", positive=("PF4", "VWF"), negative=("PTPRC",)),
    ]
    kept, dropped = _in_matrix(hypotheses, {"CD1C", "FCER1A", "CD14", "PF4", "PTPRC"})
    assert dropped == []
    assert [(item.name, item.positive, item.negative) for item in kept] == [
        ("cDC2", ("CD1C", "FCER1A"), ("CLEC9A", "CD14")),
        ("megakaryocyte", ("PF4", "VWF"), ("PTPRC",)),
    ]
    gone, dropped_all = _in_matrix(
        [Hypothesis("megakaryocyte", positive=("PF4", "VWF"), negative=("PTPRC",))],
        {"CD14"},
    )
    assert gone == []
    assert dropped_all == [
        {
            "name": "megakaryocyte",
            "reason": "missing positive markers",
            "missing": ["PF4", "VWF"],
        }
    ]


def test_report_lists_missing_genes_and_review_order():
    from annotuneagent.granularity import ontology_dir
    from annotuneagent.granularity.ontology_map import load_map
    from annotuneagent.workflow import _ontology_note

    hypotheses = [
        *_hypotheses(),
        Hypothesis("megakaryocyte", positive=("PF4", "VWF"), negative=("PTPRC",)),
    ]
    report = run(_pure(), "leiden", hypotheses=hypotheses, subcluster=False)
    assert report["dropped_hypotheses"] == [
        {
            "name": "megakaryocyte",
            "reason": "missing positive markers",
            "missing": ["PF4", "VWF"],
        }
    ]
    assert {row["cluster"] for row in report["clusters"]} == {"0", "1"}
    assert report["review"] == []
    label_map = load_map(ontology_dir() / "ontology_map.jsonl")
    pending = _ontology_note(label_map, "B cell", "human")
    assert pending["cl_id"] == "CL:0000236"
    assert pending["status"] == "proposed"
    assert pending["pending"] is True
    assert _ontology_note(label_map, "not a cell", "human") == {"status": "unmapped"}


def test_alveolar_fibroblast_is_kept_when_its_panel_is_on():
    ranking = [
        {"name": "fibroblast", "score": 0.90, "program_on": True, "program_specific": True},
        {
            "name": "alveolar fibroblast",
            "score": 0.40,
            "program_on": True,
            "program_specific": True,
        },
    ]
    settled = settle_cluster(
        marker="fibroblast",
        name="fibroblast",
        ml=MlResult("named", ("alveolar fibroblast",)),
        added=[],
        cells=1493,
        species="mouse",
        ranking=ranking,
        margin=0.10,
        marker_labeled=False,
    )
    assert settled["written_label"] == "alveolar fibroblast"
    silent = settle_cluster(
        marker="fibroblast",
        name="fibroblast",
        ml=MlResult("named", ("alveolar fibroblast",)),
        added=[],
        cells=1493,
        species="mouse",
        ranking=[
            {"name": "fibroblast", "score": 0.90, "program_on": True, "program_specific": True},
            {"name": "alveolar fibroblast", "score": 0.40, "program_on": False},
        ],
        margin=0.10,
        marker_labeled=False,
    )
    assert silent["written_label"] == "fibroblast"


def test_a_named_macrophage_subtype_is_kept_when_its_panel_is_on():
    ranking = [
        {"name": "macrophage", "score": 0.90, "program_on": True, "program_specific": True},
        {"name": "Kupffer cell", "score": 0.40, "program_on": True, "program_specific": True},
    ]
    settled = settle_cluster(
        marker="macrophage",
        name="Kupffer cell",
        ml=MlResult("named", ("Kupffer cell",)),
        added=[],
        cells=2110,
        species="human",
        ranking=ranking,
        margin=0.50,
        marker_labeled=False,
    )
    assert settled["written_label"] == "Kupffer cell"
    silent = settle_cluster(
        marker="macrophage",
        name="Kupffer cell",
        ml=MlResult("named", ("Kupffer cell",)),
        added=[],
        cells=2110,
        species="human",
        ranking=[
            {"name": "macrophage", "score": 0.90, "program_on": True, "program_specific": True},
            {"name": "Kupffer cell", "score": 0.40, "program_on": False},
        ],
        margin=0.50,
        marker_labeled=False,
    )
    assert silent["written_label"] == "macrophage"


def test_t_cell_subtype_is_not_upgraded_like_alveolar_fibroblast():
    settled = settle_cluster(
        marker="T cell",
        name="T cell",
        ml=MlResult("named", ("CD8-positive, alpha-beta T cell",)),
        added=[],
        cells=190,
        species="human",
        ranking=[
            {"name": "T cell", "score": 0.80, "program_on": True, "program_specific": True},
            {"name": "CD8-positive, alpha-beta T cell", "score": 0.40, "program_on": True},
        ],
        margin=0.40,
        marker_labeled=False,
    )
    assert settled["written_label"] == "T cell"


def test_both_cd4_and_cd8_write_t_cell():
    settled = settle_cluster(
        marker="CD4-positive, alpha-beta T cell",
        name="T cell",
        ml=MlResult("named", ("CD8-positive, alpha-beta T cell",)),
        added=[],
        cells=548,
        species="human",
        ranking=[
            {"name": "CD4-positive, alpha-beta T cell", "score": 0.70, "eligible": True},
            {"name": "CD8-positive, alpha-beta T cell", "score": 0.20, "eligible": True},
        ],
        margin=0.50,
        marker_labeled=False,
    )
    assert settled["written_label"] == "T cell"
    assert "CD4 and CD8 are both supported" in settled["hint"]
    close = settle_cluster(
        marker="naive CD4 T cell",
        name="naive T cell",
        ml=MlResult("named", ("T cell",)),
        added=[],
        cells=603,
        species="human",
        ranking=[
            {"name": "naive CD4 T cell", "score": 0.40, "eligible": True},
            {"name": "naive CD8 T cell", "score": 0.32, "eligible": True},
        ],
        margin=0.08,
        marker_labeled=False,
    )
    assert close["written_label"] == "T cell"
    clear = settle_cluster(
        marker="CD8-positive, alpha-beta T cell",
        name="CD8-positive, alpha-beta T cell",
        ml=MlResult("named", ("CD8-positive, alpha-beta T cell",)),
        added=[],
        cells=400,
        species="human",
        ranking=[
            {
                "name": "CD8-positive, alpha-beta T cell",
                "score": 1.2,
                "eligible": True,
                "program_on": True,
                "program_specific": True,
            },
            {"name": "CD4-positive, alpha-beta T cell", "score": 0.2, "eligible": True},
        ],
        margin=1.0,
        marker_labeled=True,
    )
    assert clear["written_label"] == "CD8-positive, alpha-beta T cell"


def test_mait_markers_block_effector_memory_cd8():
    settled = settle_cluster(
        marker=None,
        name="effector memory CD8 T cell",
        ml=MlResult("named", ("T cell",)),
        added=[],
        cells=437,
        species="human",
        ranking=[
            {
                "name": "effector memory CD8 T cell",
                "score": 0.55,
                "eligible": True,
                "lead_genes": ["GZMK"],
            },
            {
                "name": "MAIT cell",
                "score": 0.40,
                "eligible": True,
                "lead_genes": ["SLC4A10"],
            },
        ],
        margin=0.15,
        marker_labeled=False,
    )
    assert settled["written_label"] == "MAIT cell"
    blocked = settle_cluster(
        marker=None,
        name="effector memory CD8 T cell",
        ml=MlResult("named", ("T cell",)),
        added=[],
        cells=437,
        species="human",
        ranking=[
            {"name": "effector memory CD8 T cell", "score": 0.55, "eligible": True},
            {
                "name": "MAIT cell",
                "score": 0.10,
                "eligible": False,
                "lead_genes": ["TRAV1-2"],
            },
        ],
        margin=0.45,
        marker_labeled=False,
    )
    assert blocked["written_label"] == "T cell"


def test_nk_and_t_enriched_together_get_no_single_title():
    ranking = [
        {"name": "NK cell", "score": 1.2, "program_specific": True},
        {"name": "T cell", "score": 0.9, "program_specific": True},
        {"name": "CD8-positive, alpha-beta T cell", "score": 0.4, "program_specific": False},
    ]
    settled = settle_cluster(
        marker="T cell",
        name="T cell",
        ml=MlResult("named", ("CD8-positive, alpha-beta T cell",)),
        added=[],
        cells=1195,
        species="human",
        ranking=ranking,
        margin=0.3,
        marker_labeled=True,
    )
    assert settled["written_label"] is None
    assert settled["final_reason"] == "cytotoxic_mixture"
    blood = settle_cluster(
        marker="NK cell",
        name="effector memory CD8 T cell",
        ml=MlResult("named", ("NK cell",)),
        added=[],
        cells=1299,
        species="human",
        ranking=[
            {"name": "NK cell", "score": 1.5, "program_specific": True},
            {"name": "T cell", "score": 0.4, "program_specific": False},
        ],
        margin=1.1,
        marker_labeled=False,
    )
    assert blood["written_label"] == "NK cell"


def test_specific_mature_nkt_is_not_a_cytotoxic_mixture():
    ranking = [
        {"name": "mature NK T cell", "score": 1.4, "program_specific": True},
        {"name": "NK cell", "score": 1.1, "program_specific": True},
        {"name": "T cell", "score": 0.8, "program_specific": True},
    ]
    assert wf.cytotoxic_mixture(ranking) is False
    settled = settle_cluster(
        marker="mature NK T cell",
        name="mature NK T cell",
        ml=MlResult("named", ("mature NK T cell",)),
        added=[],
        cells=60,
        species="human",
        ranking=ranking,
        margin=0.3,
        marker_labeled=True,
    )
    assert settled["written_label"] == "mature NK T cell"
    assert settled["final_reason"] == "evidence_agreement"
    second = [
        {"name": "T cell", "score": 1.4, "program_specific": True},
        {"name": "mature NK T cell", "score": 1.2, "program_specific": True},
        {"name": "NK cell", "score": 1.0, "program_specific": True},
    ]
    assert wf.cytotoxic_mixture(second) is True
    nonspecific = [
        {"name": "mature NK T cell", "score": 1.4, "program_specific": False},
        {"name": "NK cell", "score": 1.1, "program_specific": True},
        {"name": "T cell", "score": 0.8, "program_specific": True},
    ]
    assert wf.cytotoxic_mixture(nonspecific) is True


def test_mixture_child_labels_require_every_child():
    from annotuneagent.workflow import _resolved_labels

    ranking = [
        {"name": "NK cell", "program_specific": True},
        {"name": "T cell", "program_specific": True},
    ]
    record = {"cluster": "0", "status": "label", "cell_type": "T cell", "ranking": ranking}
    partial = [
        {"cluster": "0.0", "status": "label", "cell_type": "mature NK T cell"},
        {"cluster": "0.1", "status": "uncertain", "cell_type": None},
    ]
    assert "0" not in _resolved_labels(None, "leiden", [record], {"0": partial})
    complete = [
        {"cluster": "0.0", "status": "label", "cell_type": "NK cell"},
        {"cluster": "0.1", "status": "label", "cell_type": "T cell"},
    ]
    assert _resolved_labels(None, "leiden", [record], {"0": complete})["0"] == {
        "0.0": "NK cell",
        "0.1": "T cell",
    }


def test_unsplit_cytotoxic_mixture_is_not_scored_as_its_marker_label():
    nk = ("NKG7", "GNLY", "KLRD1", "KLRF1")
    tcell = ("CD3D", "CD3E", "CD3G", "TRAC")
    bcell = ("MS4A1", "CD79A", "CD79B", "CD19", "BANK1")
    genes = [*nk, *tcell, *bcell]
    x = np.zeros((120, len(genes)), dtype=float)
    for gene in (*nk, *tcell):
        x[:80, genes.index(gene)] = 5
    for gene in bcell:
        x[80:, genes.index(gene)] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 80 + ["1"] * 40})
    adata.var_names = genes
    adata.uns["log1p"] = {"base": None}
    report = run(
        adata,
        "leiden",
        species="human",
        hypotheses=hypotheses_from_names(["NK cell", "T cell", "B cell"], "human"),
        subcluster=False,
    )
    parent = next(row for row in report["records"] if str(row["cluster"]) == "0")
    labels = adata.obs.loc[adata.obs["leiden"].astype(str) == "0", "annotune_label"]
    assert labels.isna().all()
    assert parent["status"] == "label"
    assert parent["written_label"] is None
    assert parent["final_status"] == "review"
    assert parent["final_reason"] == "cytotoxic_mixture"
    assert parent["hint"] == "NK and T programs are both enriched"
    adata.obs["author"] = np.where(adata.obs["leiden"].astype(str) == "0", "T cell", "B cell")
    summary = evaluate(report["records"], adata, "leiden", "author")
    by_cluster = {row["cluster"]: row for row in summary["clusters"]}
    assert by_cluster["0"]["status"] == "uncertain"
    assert by_cluster["0"]["exact"] is False
    assert by_cluster["1"]["exact"] is True


def test_cytotoxic_mixture_is_split_even_when_the_marker_names_it():
    nk = ("NKG7", "GNLY", "KLRD1", "KLRF1")
    tcell = ("CD3D", "CD3E", "CD3G", "TRAC")
    bcell = ("MS4A1", "CD79A", "CD79B", "CD19", "BANK1")
    genes = [*nk, *tcell, *bcell]
    x = np.zeros((120, len(genes)), dtype=float)
    for gene in nk:
        x[:80, genes.index(gene)] = 5
    for gene in tcell:
        x[:40, genes.index(gene)] = 5
    for gene in bcell:
        x[80:, genes.index(gene)] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 80 + ["1"] * 40})
    adata.var_names = genes
    adata.uns["log1p"] = {"base": None}
    report = run(
        adata,
        "leiden",
        species="human",
        hypotheses=hypotheses_from_names(["NK cell", "T cell", "B cell"], "human"),
        min_subcluster_cells=40,
    )
    labels = adata.obs.loc[adata.obs["leiden"].astype(str) == "0", "annotune_label"]
    assert not (labels.dropna().nunique() == 1 and not labels.isna().any())
    parent = next(row for row in report["records"] if str(row["cluster"]) == "0")
    assert parent.get("final_reason") in {
        "subcluster_labels",
        "cytotoxic_mixture",
        "unresolved_evidence",
    }


def test_name_and_ml_are_called():
    adata = _pure()
    calls = []

    def name(_payload):
        calls.append("name")
        return Decision("retest", ("CD4 T",))

    def ml(_payload):
        calls.append("ml")
        return MlResult("escalate")

    report = run(
        adata,
        "leiden",
        hypotheses=[_hypotheses()[1]],
        catalog=_hypotheses(),
        models=_chain(name, ml),
        model_clusters=["0"],
        subcluster=False,
    )
    assert calls == ["name"]
    cluster0 = next(row for row in report["records"] if row["cluster"] == "0")
    assert cluster0["rounds"][0]["step"] == "name"
    assert adata.obs["annotune_label"].iloc[0] != "CD8 T"


def test_user_hypothesis_replaces_same_name():
    adata = _pure()
    user = [Hypothesis("CD4 T", positive=("CD8A",), negative=("CD4",))]
    report = run(
        adata,
        "leiden",
        hypotheses=_hypotheses(),
        user_hypotheses=user,
        subcluster=False,
    )
    by_cluster = {record["cluster"]: record for record in report["records"]}
    assert by_cluster["0"]["status"] == "uncertain"


def test_one_request_enables_only_its_adapter():
    assert scales_for("name") == [
        {"id": 0, "scale": 1.0},
        {"id": 1, "scale": 0.0},
    ]
    assert scales_for("ml") == [
        {"id": 0, "scale": 0.0},
        {"id": 1, "scale": 1.0},
    ]


def test_repeated_run_clears_previous_labels():
    adata = _pure()
    hypotheses = _hypotheses()
    assert run(adata, "leiden", hypotheses=hypotheses, subcluster=False)["all_labeled"]
    report = run(adata, "leiden", hypotheses=hypotheses, min_margin=100, subcluster=False)
    assert report["written"] is False
    assert adata.obs["annotune_label"].isna().all()


def test_local_reply_that_starts_with_http_is_parsed_not_fetched(monkeypatch):
    from annotuneagent import local_lora
    from annotuneagent.workflow import _invoke

    monkeypatch.setattr(local_lora, "complete", lambda *_a, **_k: "http://not-a-url")
    fetched = []
    result, error = _invoke(
        TaskEndpoint(adapter="adapter.gguf"),
        {},
        lambda url, _payload: fetched.append(url),
        lambda text, _payload: text,
        "system",
        "name",
    )
    assert (result, error, fetched) == ("http://not-a-url", None, [])


def test_url_ml_uses_10240_when_maxlen_is_omitted():
    from annotuneagent.workflow import _invoke

    seen = {}

    def fetch(url, payload, **extra):
        seen["url"] = url
        seen.update(extra)
        return "ok"

    result, error = _invoke(
        TaskEndpoint(url="http://127.0.0.1:1"),
        {"a": 1},
        fetch,
        lambda *_a: pytest.fail("a URL endpoint must not parse local text"),
        "system",
        "ml",
    )
    assert result == "ok" and error is None
    assert seen["maxlen"] == 10240


def test_explicit_maxlen_is_kept():
    from annotuneagent.workflow import _invoke

    seen = {}

    def fetch(_url, _payload, **extra):
        seen.update(extra)
        return "ok"

    _invoke(
        TaskEndpoint(url="http://127.0.0.1:1", maxlen=2048),
        {},
        fetch,
        lambda *_a: None,
        "system",
        "ml",
    )
    assert seen["maxlen"] == 2048


def test_url_endpoint_fetches_and_never_parses():
    from annotuneagent.workflow import _invoke

    result, error = _invoke(
        TaskEndpoint(url="http://127.0.0.1:1"),
        {"a": 1},
        lambda url, payload, **_extra: (url, payload),
        lambda *_a: pytest.fail("a URL endpoint must not parse local text"),
        "system",
        "name",
    )
    assert result == ("http://127.0.0.1:1", {"a": 1}) and error is None


def test_replies_with_no_hits_escalate_instead_of_naming():
    import json

    from annotuneagent.workflow import _parse_name

    named = json.dumps({"action": "retest", "names": ["invented cell"]})
    assert _parse_name(named, {"dictionary_hits": []}).action == "escalate"


def test_ontology_names_follow_the_species_and_are_loaded_once():
    from annotuneagent.workflow import _lineage_resources

    first = _lineage_resources("human")
    assert _lineage_resources("human")[1] is first[1]
    assert first[2] is _lineage_resources("human")[2]
    assert "epithelial cell" not in first[2]
    mouse_only = {
        entry.raw_label for entry in first[1].entries if entry.species == "mouse" and entry.cl_id
    } - {
        entry.raw_label
        for entry in first[1].entries
        if entry.species in {"human", "*"} and entry.cl_id
    }
    assert mouse_only
    assert not mouse_only & first[2]
    assert mouse_only <= _lineage_resources("mouse")[2]


def test_a_voted_name_the_ontology_lacks_is_not_written(monkeypatch):
    import annotuneagent.workflow as workflow

    x = np.zeros((60, 3), dtype=float)
    x[:20, 0] = 5
    x[20:40, 1] = 5
    x[40:, 2] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20 + ["2"] * 20})
    adata.var_names = ["A", "B", "C"]
    record = {"cluster": "2", "status": "uncertain", "ranking": [], "evidence": {}}
    custom = "caller defined cell"
    monkeypatch.setattr(workflow, "panel_vote", lambda _record: custom)
    monkeypatch.setattr(workflow, "gene_vote", lambda *_args: custom)
    monkeypatch.setattr(workflow, "_neighbor_vote", lambda *_args: custom)
    resolved: dict = {}
    workflow._apply_votes(adata, "leiden", [record], {}, [], resolved, "human", None)
    assert resolved == {}
    assert record["escalate"] == "vote_name_not_in_ontology"


def test_gene_vote_ignores_a_gene_seen_in_a_handful_of_cells():
    from annotuneagent.workflow import _logfc_by_cluster

    rng = np.random.default_rng(0)
    x = np.log1p(rng.poisson(1.0, (80, 4)).astype(float))
    x[:40, 0] += 2.0  # B-like: broadly high in cluster 0
    x[:, 1] = 0.0
    x[:2, 1] = 6.0  # rare: two cells carry it, so its fold change is unstable
    adata = AnnData(X=x, obs={"leiden": ["0"] * 40 + ["1"] * 40})
    adata.var_names = ["WIDE", "RARE", "N1", "N2"]
    adata.uns["log1p"] = {"base": None}
    table = _logfc_by_cluster(adata, "leiden")
    assert table["0"]["WIDE"] > 0
    assert table["0"]["RARE"] == 0.0


def test_neighbor_vote_weights_labeled_neighbours_and_needs_a_clear_share():
    from scipy import sparse

    from annotuneagent.workflow import _neighbor_vote

    adata = AnnData(X=np.zeros((6, 1)), obs={"leiden": ["0", "0", "1", "1", "2", "2"]})
    weights = np.zeros((6, 6))
    weights[0, 2] = weights[0, 3] = 1.0  # cluster 0 cells touch cluster 1
    weights[1, 4] = 0.1  # and barely touch cluster 2
    adata.obsp["connectivities"] = sparse.csr_matrix(weights)
    labeled = {"1": "T cell", "2": "B cell"}
    assert _neighbor_vote(adata, "leiden", "0", labeled) == "T cell"
    even = np.zeros((6, 6))
    even[0, 2] = 1.0
    even[0, 4] = 1.0
    adata.obsp["connectivities"] = sparse.csr_matrix(even)
    assert (
        _neighbor_vote(adata, "leiden", "0", labeled) == "B cell"
    )  # equal share, name order breaks the tie
    assert _neighbor_vote(adata, "leiden", "0", {}) is None


def test_retest_keeps_the_votes_recorded_before_the_models_ran():
    adata = _pure()
    shared = [
        Hypothesis("CD8 T", positive=("CD3E", "CD8A"), negative=("CD4",)),
        Hypothesis("CD4 T", positive=("CD3E", "CD4"), negative=("CD8A",)),
    ]
    report = run(
        adata,
        "leiden",
        hypotheses=[shared[0]],
        catalog=shared,
        models=_chain(
            lambda _payload: Decision("retest", ("CD4 T",)),
            lambda _payload: MlResult("escalate"),
        ),
        model_clusters=["0"],
        subcluster=False,
    )
    record = next(row for row in report["records"] if row["cluster"] == "0")
    assert record["added"] == ["CD4 T"]
    assert "votes" in record


def test_cached_ontology_names_cannot_be_changed_by_a_caller():
    from annotuneagent.workflow import _lineage_resources, ontology_raw_labels

    names = ontology_raw_labels()
    assert isinstance(names, frozenset)
    assert "CD56dim NK cell" in names
    assert "Lyve1+ resident macrophage" in names
    assert isinstance(_lineage_resources("human")[2], frozenset)
    with pytest.raises(AttributeError):
        names.add("invented cell")


def test_subcluster_silhouette_is_sampled(monkeypatch):
    import numpy as np
    import scanpy as sc
    import sklearn.metrics as metrics

    seen = []
    real = metrics.silhouette_score

    def spy(*args, **kwargs):
        seen.append(args[0].shape[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(metrics, "silhouette_score", spy)
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 1, (60, 20)), rng.normal(4, 1, (60, 20))]).astype("float32")
    part = sc.AnnData(x)
    wf._subcluster(part)
    assert seen and all(size <= wf.SILHOUETTE_SAMPLE for size in seen)


def test_unclear_cluster_takes_the_title_of_one_clear_neighbor():
    from scipy import sparse

    x = np.zeros((60, 3), dtype=float)
    x[20:40, 1] = 5
    x[40:, 2] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20 + ["2"] * 20})
    adata.var_names = ["A", "B", "C"]
    adata.uns["log1p"] = {"base": None}
    weights = np.zeros((60, 60))
    weights[:20, 20:40] = 1
    weights[20:40, :20] = 1
    adata.obsp["connectivities"] = sparse.csr_matrix(weights)
    report = run(
        adata,
        "leiden",
        hypotheses=[
            Hypothesis("TypeB", positive=("B",)),
            Hypothesis("TypeC", positive=("C",)),
        ],
        subcluster=False,
    )
    by_id = {row["cluster"]: row for row in report["clusters"]}
    assert by_id["0"]["label"] == "TypeB"
    assert "near 1 TypeB" in by_id["0"]["hint"]
    assert by_id["1"]["label"] == "TypeB"
    assert by_id["2"]["label"] == "TypeC"
    assert by_id["2"]["hint"] is None


def test_a_clear_title_stays_when_another_clear_cluster_is_adjacent():
    from scipy import sparse

    x = np.zeros((40, 2), dtype=float)
    x[:20, 0] = 5
    x[20:, 1] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20})
    adata.var_names = ["A", "B"]
    adata.uns["log1p"] = {"base": None}
    weights = np.ones((40, 40))
    adata.obsp["connectivities"] = sparse.csr_matrix(weights)
    report = run(
        adata,
        "leiden",
        hypotheses=[
            Hypothesis("TypeA", positive=("A",)),
            Hypothesis("TypeB", positive=("B",)),
        ],
        subcluster=False,
    )
    by_id = {row["cluster"]: row for row in report["clusters"]}
    assert by_id["0"]["label"] == "TypeA"
    assert by_id["0"]["hint"] is None
    assert by_id["1"]["label"] == "TypeB"
    assert by_id["1"]["hint"] is None


def test_neighbor_title_keeps_a_finer_name_on_the_same_lineage():
    from annotuneagent.workflow import _apply_neighbor_titles, _lineage_resources

    dag, label_map, _names = _lineage_resources("mouse")
    adata = AnnData(X=np.zeros((4, 1)), obs={"leiden": ["4", "4", "5", "5"]})
    weights = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=float,
    )
    adata.obsp["connectivities"] = sparse.csr_matrix(weights)
    tenocyte = {
        "cluster": "4",
        "written_label": "tenocyte",
        "final_status": "named",
        "final_reason": "evidence_conflict",
        "hint": "conflict: marker smooth muscle cell",
        "rounds": [
            {"step": "name", "names": ["fibroblast"]},
            {"step": "ml", "names": ["tenocyte"]},
        ],
    }
    fibroblast = {
        "cluster": "5",
        "written_label": "fibroblast",
        "final_status": "named",
        "final_reason": "evidence_agreement",
        "hint": None,
        "rounds": [
            {"step": "name", "names": ["fibroblast"]},
            {"step": "ml", "names": ["fibroblast"]},
        ],
    }
    resolved = {"4": {"4": "tenocyte"}, "5": {"5": "fibroblast"}}
    _apply_neighbor_titles(
        adata, "leiden", [tenocyte, fibroblast], resolved, "mouse", dag, label_map
    )
    assert tenocyte["written_label"] == "tenocyte"
    assert "near 5 fibroblast" in tenocyte["hint"]
    assert resolved["4"] == {"4": "tenocyte"}


def test_neighbor_title_replaces_an_off_lineage_call_and_withdraws_a_split():
    from annotuneagent.workflow import _apply_neighbor_titles, _lineage_resources

    dag, label_map, _names = _lineage_resources("mouse")
    labels = ["6"] * 2 + ["5"] * 2 + ["7"] * 2
    adata = AnnData(X=np.zeros((6, 1)), obs={"leiden": labels})
    weights = np.zeros((6, 6))
    weights[0:2, 2:4] = 1
    weights[2:4, 0:2] = 1
    adata.obsp["connectivities"] = sparse.csr_matrix(weights)
    unclear = {
        "cluster": "6",
        "written_label": "hematopoietic stem and progenitor cell",
        "final_status": "named",
        "final_reason": "no_marker_passed",
        "hint": "no marker passed",
        "rounds": [
            {"step": "name", "action": "escalate"},
            {"step": "ml", "names": ["hematopoietic stem and progenitor cell"]},
        ],
    }
    fibroblast = {
        "cluster": "5",
        "written_label": "fibroblast",
        "final_status": "named",
        "final_reason": "evidence_agreement",
        "hint": None,
        "rounds": [
            {"step": "name", "names": ["fibroblast"]},
            {"step": "ml", "names": ["fibroblast"]},
        ],
    }
    isolated = {
        "cluster": "7",
        "written_label": "B cell",
        "final_status": "named",
        "final_reason": "evidence_agreement",
        "hint": None,
        "rounds": [
            {"step": "name", "names": ["B cell"]},
            {"step": "ml", "names": ["B cell"]},
        ],
    }
    resolved = {
        "6": {"6": unclear["written_label"]},
        "5": {"5": "fibroblast"},
        "7": {"7": "B cell"},
    }
    records = [unclear, fibroblast, isolated]
    _apply_neighbor_titles(adata, "leiden", records, resolved, "mouse", dag, label_map)
    assert unclear["written_label"] == "fibroblast"
    assert unclear["final_reason"] == "near_clear_cluster"
    assert "hematopoietic stem and progenitor cell" in unclear["hint"]
    assert "near 5 fibroblast" in unclear["hint"]
    assert "7 B cell" not in unclear["hint"]

    weights[0:2, 4:6] = 1
    weights[4:6, 0:2] = 1
    adata.obsp["connectivities"] = sparse.csr_matrix(weights)
    unclear["written_label"] = "hematopoietic stem and progenitor cell"
    unclear["final_status"] = "named"
    unclear["final_reason"] = "no_marker_passed"
    unclear["hint"] = "no marker passed"
    resolved["6"] = {"6": unclear["written_label"]}
    _apply_neighbor_titles(adata, "leiden", records, resolved, "mouse", dag, label_map)
    assert unclear["written_label"] is None
    assert unclear["final_status"] == "review"
    assert "near 5 fibroblast" in unclear["hint"]
    assert "near 7 B cell" in unclear["hint"] or "7 B cell" in unclear["hint"]
    assert "6" not in resolved


def test_umap_closeness_gives_the_clear_title_without_a_neighbor_graph():
    from annotuneagent.workflow import _apply_neighbor_titles

    coords = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.0, 0.2],
            [0.2, 0.2],
            [20.0, 20.0],
            [20.2, 20.0],
        ]
    )
    adata = AnnData(
        X=np.zeros((6, 1)),
        obs={"leiden": ["0", "0", "1", "1", "2", "2"]},
    )
    adata.obsm["X_umap"] = coords
    unclear = {
        "cluster": "0",
        "written_label": None,
        "final_status": "review",
        "final_reason": "unresolved_evidence",
        "hint": None,
    }
    close = {
        "cluster": "1",
        "written_label": "fibroblast",
        "final_status": "named",
        "final_reason": "marker_rule",
        "hint": None,
    }
    far = {
        "cluster": "2",
        "written_label": "B cell",
        "final_status": "named",
        "final_reason": "marker_rule",
        "hint": None,
    }
    resolved: dict = {}
    _apply_neighbor_titles(adata, "leiden", [unclear, close, far], resolved, None, None, None)
    assert unclear["written_label"] == "fibroblast"
    assert unclear["hint"] == "near 1 fibroblast"
    assert resolved["0"] == {"0": "fibroblast"}


def _rank_table(genes, groups):
    outside = pd.Series(0.85, index=genes)
    return {
        "params": {"groupby": "leiden"},
        "names": {group: np.array(genes) for group in groups},
        "logfoldchanges": {group: np.linspace(2, 1, len(genes)) for group in groups},
        "pts": pd.DataFrame({group: pd.Series(0.95, index=genes) for group in groups}),
        "pts_rest": pd.DataFrame({group: outside for group in groups}),
    }


def test_annotation_code_runs_after_dropping_clusters():
    import matplotlib

    matplotlib.use("Agg", force=False)
    from annotuneagent.workflow import annotation_code

    labels = ["0"] * 8 + ["1"] * 8 + ["12"] * 8
    adata = AnnData(X=np.random.default_rng(0).random((24, 4)))
    adata.obs["leiden"] = pd.Categorical(labels)
    adata.var_names = ["Cd3d", "Cd79a", "Acta2", "Tagln"]
    adata.obsm["X_umap"] = np.zeros((24, 2))
    adata.uns["log1p"] = {"base": None}
    adata.obs["annotune_label"] = adata.obs["leiden"].map(
        {"0": "T cell", "1": "B cell", "12": "granulocyte-monocyte progenitor"}
    )
    code = annotation_code(
        adata,
        [],
        "leiden",
        "annotune_label",
        {
            "T cell": ["Cd3d"],
            "B cell": ["Cd79a"],
            "granulocyte-monocyte progenitor": ["Acta2", "Tagln"],
        },
        ["T cell", "B cell", "granulocyte-monocyte progenitor"],
    )
    code = code.replace("drop_clusters = []", 'drop_clusters = ["12", 1]', 1)
    namespace = {"adata": adata}
    exec(code, namespace)
    kept = namespace["adata"]
    assert set(kept.obs["leiden"].astype(str)) == {"0"}
    assert list(kept.obs["celltype"].cat.categories) == ["T cell"]


def test_split_skips_a_cluster_whose_cells_were_removed():
    from annotuneagent.workflow import _lineage_resources, _split_lineage_disagreements

    dag, label_map, _known = _lineage_resources("mouse")
    adata = AnnData(X=np.zeros((4, 1)), obs={"leiden": ["1", "1", "2", "2"]})
    record = {
        "cluster": "12",
        "ranking": [],
        "written_label": None,
        "final_status": "removed",
        "final_reason": "low_quality",
        "rounds": [
            {"step": "name", "names": ["T cell"]},
            {"step": "ml", "names": ["B cell"]},
        ],
    }
    _split_lineage_disagreements(
        adata,
        "leiden",
        [record],
        [],
        [],
        {},
        {},
        min_margin=0.15,
        ranker=None,
        conformal_q=None,
        min_subcluster_cells=40,
        species="mouse",
        dag=dag,
        label_map=label_map,
        arbitration=[],
    )
    assert record["final_reason"] == "low_quality"


def test_half_of_the_leading_genes_counts_as_ribosomal():
    from annotuneagent.workflow import _ribosomal_signature

    def table(n_ribo: int) -> AnnData:
        genes = [f"Rps{i}" for i in range(n_ribo)] + [f"Gene{i}" for i in range(20 - n_ribo)]
        adata = AnnData(X=np.zeros((4, 1)), obs={"leiden": ["0"] * 4})
        adata.uns["rank_genes_groups"] = {
            "names": {"0": np.array(genes)},
            "logfoldchanges": {"0": np.linspace(2, 1, 20)},
            "pts_rest": pd.DataFrame({"0": pd.Series(0.85, index=genes)}),
        }
        return adata

    assert _ribosomal_signature(table(10), "0") is True
    assert _ribosomal_signature(table(9), "0") is False


def test_a_ribosomal_cluster_with_no_title_is_removed(monkeypatch):
    from annotuneagent.workflow import _drop_ribosomal_clusters, _fill_blank_labels_by_umap

    monkeypatch.setattr("annotuneagent.workflow.wilcoxon_order", lambda adata, cluster_key: {})

    ribo = [f"Rps{i}" for i in range(18)] + ["Ppia", "Ptma"]
    specific = [f"Gene{i}" for i in range(20)]
    adata = AnnData(
        X=np.zeros((30, 1)),
        obs={"leiden": ["12"] * 10 + ["14"] * 10 + ["13"] * 10},
    )
    adata.var_names = ["G"]
    adata.uns["log1p"] = {"base": None}
    adata.obsm["X_umap"] = np.vstack(
        [np.zeros((10, 2)), np.full((10, 2), 1.0), np.full((10, 2), 0.04)]
    )
    adata.uns["rank_genes_groups"] = _rank_table(ribo, ["12", "13"])
    specific_table = _rank_table(specific, ["14"])
    for key in ("names", "logfoldchanges"):
        adata.uns["rank_genes_groups"][key]["14"] = specific_table[key]["14"]
    adata.uns["rank_genes_groups"]["pts"]["14"] = specific_table["pts"]["14"]
    adata.uns["rank_genes_groups"]["pts_rest"]["14"] = specific_table["pts_rest"]["14"]
    blank = {
        "cluster": "12",
        "written_label": None,
        "final_status": "out_of_dictionary",
        "final_reason": "out_of_dictionary",
        "hint": "unexplained: Ctla2a",
    }
    other_blank = {
        "cluster": "14",
        "written_label": None,
        "final_status": "out_of_dictionary",
        "final_reason": "out_of_dictionary",
        "hint": "unexplained: Ctla2a",
    }
    named = {
        "cluster": "13",
        "written_label": "granulocyte-monocyte progenitor",
        "final_status": "named",
        "final_reason": "evidence_agreement",
        "hint": None,
    }
    resolved = {"13": {"13": "granulocyte-monocyte progenitor"}}
    _drop_ribosomal_clusters(adata, "leiden", [blank, other_blank, named], resolved)
    assert blank["final_reason"] == "low_quality"
    assert blank["hint"] == "ribosomal genes only"
    assert blank["cells_removed"] == 10
    assert "Enriched versus other cells:" in blank["query"]
    assert "Rps0" in blank["query"]
    assert "10 cells" in blank["query"]
    assert "12" not in resolved
    assert other_blank["final_reason"] == "out_of_dictionary"
    assert other_blank["written_label"] is None
    assert named["written_label"] == "granulocyte-monocyte progenitor"
    assert set(adata.obs["leiden"].astype(str)) == {"14", "13"}
    _fill_blank_labels_by_umap(
        adata, "leiden", [blank, other_blank, named], resolved, None, None, None
    )
    assert blank["written_label"] is None
    saved = blank["query"]
    wf._attach_queries(adata, "leiden", [blank], [], None, None)
    assert blank["query"] == saved


def test_a_blank_cluster_beside_one_lineage_takes_that_name():
    from annotuneagent.workflow import _fill_blank_labels_by_umap

    coords = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.4, 0.0],
            [0.6, 0.0],
            [8.0, 8.0],
            [8.2, 8.0],
        ]
    )
    adata = AnnData(
        X=np.zeros((6, 1)),
        obs={"leiden": ["0", "0", "1", "1", "2", "2"]},
    )
    adata.obsm["X_umap"] = coords
    blank = {
        "cluster": "0",
        "written_label": None,
        "final_status": "review",
        "final_reason": "unresolved_evidence",
        "hint": None,
    }
    neutrophil = {
        "cluster": "1",
        "written_label": "neutrophil",
        "final_status": "review",
        "final_reason": "weak_agreement",
        "hint": "weak agreement on neutrophil",
    }
    far = {
        "cluster": "2",
        "written_label": "B cell",
        "final_status": "named",
        "final_reason": "evidence_agreement",
        "hint": None,
    }
    resolved = {"1": {"1": "neutrophil"}, "2": {"2": "B cell"}}
    _fill_blank_labels_by_umap(
        adata, "leiden", [blank, neutrophil, far], resolved, None, None, None
    )
    assert blank["written_label"] == "neutrophil"
    assert blank["final_reason"] == "near_clear_cluster"
    assert "near 1 neutrophil" in blank["hint"]
    assert "umap distance" in blank["hint"]
    assert resolved["0"] == {"0": "neutrophil"}

    between = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [1.0, 0.0],
            [1.2, 0.0],
            [0.45, 0.0],
            [0.65, 0.0],
        ]
    )
    mixed = AnnData(
        X=np.zeros((6, 1)),
        obs={"leiden": ["1", "1", "2", "2", "3", "3"]},
    )
    mixed.obsm["X_umap"] = between
    middle = {
        "cluster": "3",
        "written_label": None,
        "final_status": "review",
        "final_reason": "unresolved_evidence",
        "hint": None,
    }
    kept = {"1": {"1": "neutrophil"}, "2": {"2": "B cell"}}
    _fill_blank_labels_by_umap(mixed, "leiden", [middle, neutrophil, far], kept, None, None, None)
    assert middle["written_label"] == "neutrophil"
    assert "near 1 neutrophil" in middle["hint"]
    assert kept["3"] == {"3": "neutrophil"}

    withheld = {
        "cluster": "1",
        "written_label": "endothelial cell",
        "final_status": "out_of_dictionary",
        "final_reason": "out_of_dictionary",
        "hint": "unexplained: Emcn",
    }
    untouched = {
        "cluster": "0",
        "written_label": None,
        "final_status": "out_of_dictionary",
        "final_reason": "out_of_dictionary",
        "hint": "unexplained: Emcn",
    }
    blocked: dict = {}
    _fill_blank_labels_by_umap(
        adata, "leiden", [untouched, withheld, far], blocked, None, None, None
    )
    assert untouched["written_label"] is None
    assert "0" not in blocked


def test_two_names_are_not_split_into_more_than_two_clusters(monkeypatch):
    import scanpy as sc

    part = AnnData(np.random.default_rng(0).normal(size=(141, 6)))

    def fake_pca(adata, **_kwargs):
        adata.obsm["X_pca"] = np.asarray(adata.X)[:, :2]

    def fake_leiden(adata, **kwargs):
        if kwargs["resolution"] >= 0.2:
            labels = np.array(["0"] * 40 + ["1"] * 40 + ["2"] * 40 + ["3"] * 21)
        else:
            labels = np.array(["0"] * 70 + ["1"] * 71)
        adata.obs["annotune_sub_try"] = labels

    monkeypatch.setattr(sc.pp, "pca", fake_pca)
    monkeypatch.setattr(sc.pp, "neighbors", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sc.tl, "leiden", fake_leiden)
    assert wf._subcluster(part, max_clusters=2) is True
    assert part.obs[wf.SUBCLUSTER_KEY].nunique() == 2


def test_a_cluster_of_141_cells_is_not_tried_at_resolution_0_6():
    assert wf._subcluster_resolutions(141) == (0.1, 0.2)
    assert max(wf._subcluster_resolutions(141)) <= 0.2
    assert wf._subcluster_resolutions(300) == (0.2, 0.4)
    assert wf._subcluster_resolutions(3000) == wf.SUBCLUSTER_RESOLUTIONS


def test_subcluster_sample_keeps_a_rare_group(monkeypatch):
    import scanpy as sc
    from sklearn.utils import check_random_state

    n_cells = 3000
    sampled_before = set(check_random_state(0).permutation(n_cells)[: wf.SILHOUETTE_SAMPLE])
    rare_cell = next(index for index in range(n_cells) if index not in sampled_before)
    part = sc.AnnData(np.random.default_rng(1).normal(size=(n_cells, 3)))

    def fake_pca(adata, **_kwargs):
        adata.obsm["X_pca"] = np.asarray(adata.X)[:, :2]

    def fake_leiden(adata, **_kwargs):
        labels = np.zeros(adata.n_obs, dtype=int)
        labels[rare_cell] = 1
        adata.obs["annotune_sub_try"] = labels.astype(str)

    monkeypatch.setattr(sc.pp, "pca", fake_pca)
    monkeypatch.setattr(sc.pp, "neighbors", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sc.tl, "leiden", fake_leiden)
    monkeypatch.setattr(wf, "SUBCLUSTER_RESOLUTIONS", (0.6,))
    assert wf._subcluster(part) is True
    assert part.obs[wf.SUBCLUSTER_KEY].iloc[rare_cell] == "1"
