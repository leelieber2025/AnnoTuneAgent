import numpy as np
from anndata import AnnData

from annotuneagent import Hypothesis, annotate, evaluate, hypotheses_from_names
from annotuneagent.markers import parents_for


def _labeled():
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
    return adata


def test_exact_and_parent_match():
    adata = _labeled()
    names = ["CD4-positive, alpha-beta T cell", "CD8-positive, alpha-beta T cell"]
    records = annotate(adata, "leiden", hypotheses_from_names(names, "human"))
    summary = evaluate(
        records, adata, "leiden", "author", parents=parents_for(names, species="human")
    )
    assert summary["n_exact"] == 2
    assert summary["n_coarse"] == 2
    assert summary["n_uncertain"] == 0

    coarse_only = [
        Hypothesis("T cell", positive=("CD3E",), negative=("MS4A1",)),
        Hypothesis("B cell", positive=("MS4A1",), negative=("CD3E",)),
    ]
    background = np.zeros((20, adata.n_vars), dtype=float)
    background[:, list(adata.var_names).index("MS4A1")] = 5
    mixed = AnnData(
        X=np.vstack([np.asarray(adata.X), background]),
        obs={
            "leiden": list(adata.obs["leiden"]) + ["2"] * 20,
            "author": list(adata.obs["author"]) + ["B cell"] * 20,
        },
    )
    mixed.var_names = adata.var_names
    mixed.uns["log1p"] = {"base": None}
    records = annotate(mixed, "leiden", coarse_only)
    adata = mixed
    parents = {
        "CD4-positive, alpha-beta T cell": "T cell",
        "CD8-positive, alpha-beta T cell": "T cell",
    }
    summary = evaluate(records, adata, "leiden", "author", parents=parents)
    by_cluster = {item["cluster"]: item for item in summary["clusters"]}
    assert by_cluster["0"]["exact"] is False
    assert by_cluster["1"]["exact"] is False
    assert by_cluster["0"]["coarse"] is True
    assert by_cluster["1"]["coarse"] is True
    assert by_cluster["0"]["status"] == "label"


def test_uncertain_is_not_a_match():
    adata = _labeled()
    adata.X[20:, list(adata.var_names).index("CD8A")] = 0
    adata.X[:20, list(adata.var_names).index("CD8A")] = 6
    names = ["CD4-positive, alpha-beta T cell", "CD8-positive, alpha-beta T cell"]
    records = annotate(adata, "leiden", hypotheses_from_names(names, "human"))
    summary = evaluate(records, adata, "leiden", "author", parents=parents_for(species="human"))
    cluster0 = next(item for item in summary["clusters"] if item["cluster"] == "0")
    assert cluster0["status"] == "uncertain"
    assert cluster0["exact"] is False
    assert summary["n_uncertain"] == 1


def test_tied_reference_is_ambiguous():
    adata = _labeled()
    adata.obs["author"] = ["CD4-positive, alpha-beta T cell"] * 10 + [
        "CD8-positive, alpha-beta T cell"
    ] * 30
    records = annotate(
        adata,
        "leiden",
        hypotheses_from_names(
            ["CD4-positive, alpha-beta T cell", "CD8-positive, alpha-beta T cell"],
            "human",
        ),
    )
    summary = evaluate(records, adata, "leiden", "author")
    assert summary["n_ambiguous_reference"] == 1
    assert summary["n_exact"] == 1


def test_unknown_dictionary_name():
    try:
        hypotheses_from_names(["not-a-type"], "human")
    except KeyError as exc:
        assert "not-a-type" in str(exc)
    else:
        raise AssertionError("expected KeyError")


def test_workflow_evaluation_uses_written_labels():
    adata = _labeled()
    records = annotate(
        adata,
        "leiden",
        hypotheses_from_names(
            ["CD4-positive, alpha-beta T cell", "CD8-positive, alpha-beta T cell"], "human"
        ),
    )
    adata.obs["annotune_label"] = adata.obs["author"].copy()
    records[0]["cell_type"] = "wrong original result"
    summary = evaluate(records, adata, "leiden", "author", prediction_key="annotune_label")
    assert summary["n_exact"] == 2
    assert summary["clusters"][0]["predicted"] == "CD4-positive, alpha-beta T cell"


def test_missing_cluster_labels_are_an_error_not_a_cluster_called_nan():
    import pytest

    adata = _labeled()
    adata.obs["leiden"] = adata.obs["leiden"].astype("category")
    adata.obs.loc[adata.obs_names[:3], "leiden"] = np.nan
    names = ["CD4-positive, alpha-beta T cell", "CD8-positive, alpha-beta T cell"]
    with pytest.raises(ValueError, match="3 cells with no cluster"):
        annotate(adata, "leiden", hypotheses_from_names(names, "human"))


def test_unmapped_majority_is_not_replaced_by_a_mapped_minority():
    from annotuneagent.evaluate import cluster_reference

    raw = ["placental villous trophoblast"] * 80 + ["extravillous trophoblast"] * 20
    mapped = [None] * 80 + ["extravillous trophoblast"] * 20
    assert cluster_reference(raw, mapped) is None
    pooled_raw = ["FC"] * 30 + ["RegC"] * 30 + ["fibroblast"] * 40
    pooled_mapped = ["chondrocyte"] * 60 + ["fibroblast"] * 40
    assert cluster_reference(pooled_raw, pooled_mapped) == "chondrocyte"
    blank = [None] * 15 + ["CD4-positive, alpha-beta T cell"] * 5
    assert cluster_reference(blank, blank) == "CD4-positive, alpha-beta T cell"


def test_evaluate_does_not_score_a_mapped_minority():
    adata = _labeled()
    raw = ["placental villous trophoblast"] * 15 + ["CD4-positive, alpha-beta T cell"] * 5
    raw += ["CD8-positive, alpha-beta T cell"] * 20
    mapped = [None] * 15 + ["CD4-positive, alpha-beta T cell"] * 5
    mapped += ["CD8-positive, alpha-beta T cell"] * 20
    adata.obs["author_raw"] = raw
    adata.obs["author"] = mapped
    records = [
        {"cluster": "0", "status": "label", "cell_type": "CD4-positive, alpha-beta T cell"},
        {"cluster": "1", "status": "label", "cell_type": "CD8-positive, alpha-beta T cell"},
    ]
    summary = evaluate(records, adata, "leiden", "author", raw_label_key="author_raw")
    by_cluster = {row["cluster"]: row for row in summary["clusters"]}
    assert by_cluster["0"]["reference"] is None
    assert by_cluster["0"]["exact"] is False
    assert by_cluster["1"]["reference"] == "CD8-positive, alpha-beta T cell"
    assert summary["n_exact"] == 1
    assert summary["n_ambiguous_reference"] == 1


def test_cells_without_an_author_label_take_no_part_in_the_majority():
    adata = _labeled()
    names = ["CD4-positive, alpha-beta T cell", "CD8-positive, alpha-beta T cell"]
    records = annotate(adata, "leiden", hypotheses_from_names(names, "human"))
    author = adata.obs["author"].astype(object)
    author.iloc[:15] = np.nan
    adata.obs["author"] = author
    result = evaluate(records, adata, "leiden", "author")
    by_cluster = {row["cluster"]: row for row in result["clusters"]}
    assert by_cluster["0"]["reference"] == "CD4-positive, alpha-beta T cell"
    assert result["n_exact"] == 2
