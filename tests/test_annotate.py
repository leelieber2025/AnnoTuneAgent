import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse, stats

from annotuneagent import Hypothesis, annotate
from annotuneagent.evidence import cluster_mask, marker_statistics, undeclared_enriched


def _adata():
    # Cluster 0: CD3E+ CD4+. Cluster 1: CD3E+ CD8A+.
    x = np.zeros((40, 4), dtype=float)
    x[:20, 0] = 5  # CD3E
    x[:20, 1] = 5  # CD4
    x[20:, 0] = 5
    x[20:, 2] = 5  # CD8A
    obs_cluster = ["0"] * 20 + ["1"] * 20
    adata = AnnData(X=x, obs={"leiden": obs_cluster})
    adata.var_names = ["CD3E", "CD4", "CD8A", "IL7R"]
    adata.uns["log1p"] = {"base": None}
    return adata


def _hypotheses():
    return [
        Hypothesis("CD4 T", positive=("CD3E", "CD4"), negative=("CD8A",)),
        Hypothesis("CD8 T", positive=("CD3E", "CD8A"), negative=("CD4",)),
    ]


def test_separates_two_clusters():
    records = annotate(_adata(), "leiden", _hypotheses())
    by_cluster = {record["cluster"]: record for record in records}
    assert by_cluster["0"]["status"] == "label"
    assert by_cluster["0"]["cell_type"] == "CD4 T"
    assert by_cluster["1"]["cell_type"] == "CD8 T"
    assert by_cluster["0"]["margin"] > 0
    assert "CD8A" in by_cluster["0"]["reasoning"] or "CD4" in by_cluster["0"]["reasoning"]


def test_support_gene_adds_only_when_enriched():
    from annotuneagent.arbitrate import score_hypothesis
    from annotuneagent.evidence import GeneEvidence

    def gene(name: str, logfc: float) -> GeneEvidence:
        return GeneEvidence(
            gene=name,
            direction="positive",
            pct_in=0.8,
            pct_out=0.1,
            mean_in=1.0,
            mean_out=0.1,
            logfc=logfc,
            pvalue=0.01,
            expressed=logfc > 0,
        )

    hypothesis = Hypothesis("CD4 T", positive=("CD3E", "IL7R"), support=("CD4",))
    quiet = score_hypothesis(
        hypothesis, [gene("CD3E", 2.0), gene("IL7R", 2.0)], [], [gene("CD4", 0.0)]
    )
    boosted = score_hypothesis(
        hypothesis, [gene("CD3E", 2.0), gene("IL7R", 2.0)], [], [gene("CD4", 0.4)]
    )
    assert quiet.score == pytest.approx(2.0)
    assert boosted.score == pytest.approx(2.4)


def test_undetected_positive_stays_in_the_mean():
    from annotuneagent.arbitrate import score_hypothesis
    from annotuneagent.evidence import GeneEvidence

    def gene(name: str, logfc: float, pct_in: float) -> GeneEvidence:
        return GeneEvidence(
            gene=name,
            direction="positive",
            pct_in=pct_in,
            pct_out=0.1,
            mean_in=1.0,
            mean_out=0.1,
            logfc=logfc,
            pvalue=0.01,
            expressed=pct_in >= 0.25,
        )

    hypothesis = Hypothesis("pancreatic delta cell", positive=("Sst", "Hhex", "Ghsr"))
    scored = score_hypothesis(
        hypothesis,
        [gene("Sst", 9.0, 1.0), gene("Hhex", 2.0, 0.8), gene("Ghsr", 0.0, 0.03)],
        [],
    )
    # Ghsr is undetected, so it contributes 0. The mean is (9 + 2 + 0) / 3.
    assert scored.score == pytest.approx(11 / 3)


def test_one_detected_gene_does_not_stand_for_the_panel():
    from annotuneagent.arbitrate import score_hypothesis
    from annotuneagent.evidence import GeneEvidence

    def gene(name: str, logfc: float, pct_in: float) -> GeneEvidence:
        return GeneEvidence(
            gene=name,
            direction="positive",
            pct_in=pct_in,
            pct_out=0.01,
            mean_in=1.0,
            mean_out=0.0,
            logfc=logfc,
            pvalue=0.01,
            expressed=pct_in >= 0.25,
        )

    hypothesis = Hypothesis(
        "CD56dim NK cell",
        positive=("FCGR3A", "FGFBP2", "GZMB", "PRF1", "SPON2"),
    )
    scored = score_hypothesis(
        hypothesis,
        [
            gene("FCGR3A", 0.0, 0.03),
            gene("FGFBP2", 0.0, 0.02),
            gene("GZMB", 4.5, 0.99),
            gene("PRF1", 0.0, 0.01),
            gene("SPON2", 0.0, 0.14),
        ],
        [],
    )
    assert scored.score == pytest.approx(0.9)


def test_absent_genes_stay_in_the_mean():
    from annotuneagent.arbitrate import score_hypothesis
    from annotuneagent.evidence import GeneEvidence

    def gene(name: str, logfc: float, pct_in: float) -> GeneEvidence:
        return GeneEvidence(
            gene=name,
            direction="positive",
            pct_in=pct_in,
            pct_out=0.1,
            mean_in=1.0,
            mean_out=0.1,
            logfc=logfc,
            pvalue=0.01,
            expressed=pct_in >= 0.25,
        )

    hypothesis = Hypothesis(
        "endothelial cell",
        positive=("PECAM1", "VWF", "CDH5", "CLDN5", "KDR"),
    )
    scored = score_hypothesis(
        hypothesis,
        [
            gene("PECAM1", 4.0, 0.9),
            gene("VWF", 4.0, 0.8),
            gene("CDH5", 0.0, 0.02),
            gene("CLDN5", 0.0, 0.01),
            gene("KDR", 0.0, 0.01),
        ],
        [],
    )
    # Three undetected genes stay in the mean as 0: (4 + 4 + 0 + 0 + 0) / 5.
    assert scored.score == pytest.approx(1.6)


def test_absent_identity_gene_dilutes_shared_markers():
    from annotuneagent.arbitrate import score_hypothesis
    from annotuneagent.evidence import GeneEvidence

    def gene(name: str, logfc: float, pct_in: float) -> GeneEvidence:
        return GeneEvidence(
            gene=name,
            direction="positive",
            pct_in=pct_in,
            pct_out=0.1,
            mean_in=1.0,
            mean_out=0.1,
            logfc=logfc,
            pvalue=0.01,
            expressed=pct_in >= 0.25,
        )

    # Mouse memory CD4 is Cd4, S100a4, Cd44. Monocytes express the last two.
    # A high fold change on an undetected Cd4 still contributes 0.
    hypothesis = Hypothesis("memory CD4 T cell", positive=("Cd4", "S100a4", "Cd44"))
    scored = score_hypothesis(
        hypothesis,
        [gene("Cd4", 5.0, 0.0), gene("S100a4", 4.0, 0.9), gene("Cd44", 2.0, 0.8)],
        [],
    )
    assert scored.score == pytest.approx(2.0)


def test_one_negative_gene_weighs_like_one_positive():
    from annotuneagent.arbitrate import score_hypothesis
    from annotuneagent.evidence import GeneEvidence

    def gene(name: str, logfc: float, direction: str) -> GeneEvidence:
        return GeneEvidence(
            gene=name,
            direction=direction,
            pct_in=0.8,
            pct_out=0.1,
            mean_in=1.0,
            mean_out=0.1,
            logfc=logfc,
            pvalue=0.01,
            expressed=True,
        )

    hypothesis = Hypothesis(
        "plasmacytoid dendritic cell",
        positive=("Siglech", "Ccr9", "Ly6d", "Cox6a2", "Bst2"),
        negative=("Cd209a",),
    )
    scored = score_hypothesis(
        hypothesis,
        [
            gene("Siglech", 4.0, "positive"),
            gene("Ccr9", 3.0, "positive"),
            gene("Ly6d", 3.0, "positive"),
            gene("Cox6a2", 3.0, "positive"),
            gene("Bst2", 2.0, "positive"),
        ],
        [gene("Cd209a", 2.0, "negative")],
    )
    # Mean of the five positives is 3. The opposing gene subtracts 2/5.
    assert scored.score == pytest.approx(2.6)
    assert scored.penalty == pytest.approx(0.4)


def test_t_cell_descendant_without_a_receptor_loses_to_monocyte():
    from annotuneagent.markers import hypotheses_from_names

    genes = [
        "Cd4",
        "S100a4",
        "Cd44",
        "Cd8b1",
        "Foxp3",
        "Cd3d",
        "Cd3e",
        "Cd3g",
        "Trac",
        "Lyz2",
        "Cd14",
        "Csf1r",
        "Ms4a1",
        "Ly6g",
    ]
    x = np.zeros((40, len(genes)), dtype=float)
    # Cluster 0: macrophage genes plus the shared memory-CD4 genes. No Cd3.
    for gene in ("S100a4", "Cd44", "Lyz2", "Cd14", "Csf1r"):
        x[:20, genes.index(gene)] = 5
    # Cluster 1: memory CD4 with Cd3e.
    for gene in ("Cd3e", "Cd4", "S100a4", "Cd44"):
        x[20:, genes.index(gene)] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20})
    adata.var_names = genes
    adata.uns["log1p"] = {"base": None}
    records = annotate(
        adata,
        "leiden",
        hypotheses_from_names(["memory CD4 T cell", "monocyte"], "mouse"),
    )
    by_cluster = {record["cluster"]: record for record in records}
    assert by_cluster["0"]["cell_type"] == "monocyte"
    memory = next(row for row in by_cluster["0"]["ranking"] if row["name"] == "memory CD4 T cell")
    assert memory["eligible"] is False
    assert by_cluster["1"]["cell_type"] == "memory CD4 T cell"


def test_mait_markers_penalize_memory_cd4():
    from annotuneagent.markers import hypotheses_from_names

    genes = [
        "IL7R",
        "S100A4",
        "ANXA1",
        "KLRB1",
        "CD8B",
        "FOXP3",
        "SLC4A10",
        "TRAV1-2",
        "CD3D",
        "CD3E",
        "CD3G",
        "TRAC",
        "ZBTB16",
        "RORC",
        "CD4",
        "TRDC",
    ]
    x = np.zeros((40, len(genes)), dtype=float)
    for gene in ("CD3D", "CD3E", "IL7R", "KLRB1", "SLC4A10", "TRAV1-2", "ZBTB16", "RORC"):
        x[:20, genes.index(gene)] = 5
    x[20:, genes.index("ANXA1")] = 5
    adata = AnnData(X=x, obs={"leiden": ["0"] * 20 + ["1"] * 20})
    adata.var_names = genes
    adata.uns["log1p"] = {"base": None}
    records = annotate(
        adata,
        "leiden",
        hypotheses_from_names(["memory CD4 T cell", "MAIT cell"], "human"),
    )
    by_cluster = {record["cluster"]: record for record in records}
    assert by_cluster["0"]["cell_type"] == "MAIT cell"


def test_one_detected_gene_is_not_a_program():
    from annotuneagent.arbitrate import program_detected
    from annotuneagent.evidence import GeneEvidence

    def gene(pct: float) -> GeneEvidence:
        return GeneEvidence(
            gene="G",
            direction="positive",
            pct_in=pct,
            pct_out=0.1,
            mean_in=1.0,
            mean_out=0.1,
            logfc=1.0,
            pvalue=0.01,
            expressed=True,
        )

    assert program_detected([gene(1.0), gene(0.1)]) is False
    assert program_detected([gene(1.0), gene(0.8)]) is True
    assert program_detected([gene(1.0), gene(0.1), gene(0.1), gene(0.1)]) is False
    assert program_detected([gene(1.0), gene(0.8), gene(0.1), gene(0.1)]) is True


def test_negative_marker_blocks_label():
    adata = _adata()
    adata.X[20:, 2] = 0
    adata.X[:20, 2] = 4  # CD8A enriched in the CD4 cluster, not in the background
    records = annotate(adata, "leiden", _hypotheses())
    cluster0 = next(record for record in records if record["cluster"] == "0")
    assert cluster0["status"] == "uncertain"
    assert cluster0["cell_type"] is None


def test_relative_enrichment_counts_when_other_cells_also_express():
    adata = _adata()
    adata.X[:, 0] = 0.4
    adata.X[:20, 0] = 3
    adata.X[:, 1] = 0.4
    adata.X[:20, 1] = 3
    records = annotate(adata, "leiden", _hypotheses())
    cluster0 = next(record for record in records if record["cluster"] == "0")
    assert cluster0["status"] == "label"
    assert cluster0["cell_type"] == "CD4 T"


def test_ambient_negative_does_not_block():
    adata = _adata()
    adata.X[:, 2] = 1  # CD8A at the same level in every cell
    records = annotate(adata, "leiden", _hypotheses())
    cluster0 = next(record for record in records if record["cluster"] == "0")
    assert cluster0["status"] == "label"
    assert cluster0["cell_type"] == "CD4 T"


def test_missing_gene_raises():
    try:
        annotate(
            _adata(),
            "leiden",
            [Hypothesis("NK", positive=("NCAM1",), negative=())],
        )
    except KeyError as exc:
        assert "NCAM1" in str(exc)
    else:
        raise AssertionError("expected KeyError")


def test_single_candidate_without_positive_evidence_is_uncertain():
    adata = _adata()
    adata.X[:] = 0
    records = annotate(adata, "leiden", [Hypothesis("T cell", positive=("CD3E",))])
    assert all(row["status"] == "uncertain" and row["cell_type"] is None for row in records)


def test_log1p_input_is_required_and_fold_change_uses_linear_expression():
    adata = _adata()
    adata.X[:] = 0
    adata.X[:20, 1] = np.log1p(9)
    del adata.uns["log1p"]
    record = annotate(adata, "leiden", [Hypothesis("CD4 T", positive=("CD4",))])[0]
    assert adata.uns["log1p"] == {"base": None}
    evidence = record["evidence"]["positive"][0]
    assert evidence["mean_in"] == pytest.approx(9)
    assert evidence["logfc"] == pytest.approx(np.log2(10))
    raw = _adata()
    raw.X[:] = 0
    raw.X[:20, 1] = 9
    del raw.uns["log1p"]
    with pytest.raises(ValueError, match="normalize_total and log1p"):
        annotate(raw, "leiden", [Hypothesis("CD4 T", positive=("CD4",))])


def test_batch_measurements_match_direct_mannwhitney():
    rng = np.random.default_rng(1)
    x = rng.random((60, 5))
    x[x < 0.55] = 0
    x[:, 2] = 0.4
    x[:, 4] = np.linspace(0.2, 2.5, 60)
    adata = AnnData(
        X=sparse.csr_matrix(x),
        obs={"leiden": ["0"] * 5 + ["1"] * 25 + ["2"] * 30},
    )
    adata.var_names = ["G0", "G1", "G2", "G3", "G4"]
    adata.uns["log1p"] = {"base": None}
    genes = list(adata.var_names)
    clusters = ["0", "1", "2"]
    masks = [cluster_mask(adata, "leiden", cluster) for cluster in clusters]
    measured = marker_statistics(adata, masks, genes)
    dense = np.asarray(adata.X.toarray())
    for cluster, mask in enumerate(masks):
        for index, gene in enumerate(genes):
            inside = dense[mask, index]
            outside = dense[~mask, index]
            pct_in, pct_out, mean_in, mean_out, logfc, pvalue = measured[cluster][gene]
            assert pct_in == pytest.approx(float(np.mean(inside > 0)))
            assert pct_out == pytest.approx(float(np.mean(outside > 0)))
            assert mean_in == pytest.approx(float(np.mean(np.expm1(inside))))
            assert mean_out == pytest.approx(float(np.mean(np.expm1(outside))))
            assert logfc == pytest.approx(float(np.log2((mean_in + 1.0) / (mean_out + 1.0))))
            if (
                np.unique(inside).size == 1
                and np.unique(outside).size == 1
                and inside[0] == outside[0]
            ):
                expected = 1.0
            else:
                expected = float(
                    stats.mannwhitneyu(inside, outside, alternative="two-sided").pvalue
                )
            assert pvalue == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_undeclared_genes_match_a_per_cluster_mean():
    rng = np.random.default_rng(2)
    x = rng.random((40, 6))
    x[x < 0.7] = 0
    x[:20, 5] = 3
    matrix = sparse.csr_matrix(x)
    adata = AnnData(X=matrix.copy(), obs={"leiden": ["0"] * 20 + ["1"] * 20})
    adata.var_names = ["G0", "G1", "G2", "G3", "G4", "G5"]
    adata.uns["log1p"] = {"base": None}
    mask = cluster_mask(adata, "leiden", "0")
    found = undeclared_enriched(adata, mask, {"G0"})
    reference = _reference_undeclared(matrix, mask, {"G0"}, list(adata.var_names))
    assert found == reference
    assert found[0] == "G5"


def _reference_undeclared(matrix, mask, declared, names):
    def group_mean(selected):
        block = matrix[selected].tocsr()
        linear = block.copy()
        linear.data = np.expm1(linear.data)
        return np.asarray(linear.mean(axis=0)).ravel()

    mean_in = group_mean(mask)
    mean_out = group_mean(~mask)
    logfc = np.log2((mean_in + 1.0) / (mean_out + 1.0))
    keep = logfc >= 1.0
    for index, name in enumerate(names):
        if name in declared:
            keep[index] = False
    chosen = []
    for index in np.argsort(-logfc):
        if not keep[index]:
            continue
        chosen.append(names[index])
        if len(chosen) >= 10:
            break
    return chosen


def test_clusters_option_returns_the_same_record_for_that_cluster():
    adata = _adata()
    everything = {row["cluster"]: row for row in annotate(adata, "leiden", _hypotheses())}
    only = annotate(adata, "leiden", _hypotheses(), clusters=["1"])
    assert [row["cluster"] for row in only] == ["1"]
    assert only[0] == everything["1"]


def test_dense_cluster_sums_match_when_split_into_column_blocks(monkeypatch):
    from annotuneagent import evidence

    rng = np.random.default_rng(1)
    x = np.log1p(rng.poisson(0.3, (60, 9)).astype(np.float32))
    codes = rng.integers(-1, 3, 60).astype(np.int32)
    monkeypatch.setattr(evidence, "_MARKER_CHUNK_BYTES", 8 * 60 * 2)  # two genes per block
    sums, totals = evidence._group_linear_sums(x, codes, 3)
    linear = np.expm1(x.astype(np.float64))
    expected = np.stack([linear[codes == cluster].sum(axis=0) for cluster in range(3)])
    assert np.allclose(sums, expected)
    assert np.allclose(totals, linear.sum(axis=0))


def test_numba_kernels_release_the_gil():
    from annotuneagent import evidence

    assert evidence._marker_chunk.targetoptions["nogil"]
    assert evidence._sum_row_range.targetoptions["nogil"]
