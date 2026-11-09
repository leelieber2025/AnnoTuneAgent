import numpy as np
import pytest
from anndata import AnnData
from scipy import stats

from annotuneagent.prepare import (
    RAW_REQUIRED,
    _align_scrublet_threshold,
    _denoise,
    _drop_doublets,
    _drop_solo_doublets,
    _expected_doublet_rate,
    _filter_qc,
    _format_filter_report,
    _on_count_scale,
    annotation_resolution,
    elbow_count,
    prepare,
)


def _counts():
    rng = np.random.default_rng(0)
    x = rng.poisson(1, size=(80, 40)).astype(np.float64)
    x[:40, :12] += 25
    x[40:, 12:24] += 25
    adata = AnnData(X=x)
    adata.var_names = [f"Gene{i}" for i in range(adata.n_vars)]
    adata.obs_names = [f"cell{i}" for i in range(adata.n_obs)]
    return adata


def test_singular_batch_hvg_is_retried_without_the_batch(monkeypatch):
    calls = []

    def fake(adata, n_top_genes, flavor, batch_key):
        calls.append(batch_key)
        if batch_key is not None:
            raise ValueError("singular loess fit")
        adata.var["highly_variable"] = [index < 10 for index in range(adata.n_vars)]

    monkeypatch.setattr("scanpy.pp.highly_variable_genes", fake)
    adata = _counts()
    adata.obs["donor"] = ["a"] * 40 + ["b"] * 40
    prepare(adata, species="mouse", batch_key="donor", show=False)
    assert calls == ["donor", None]


@pytest.mark.parametrize(
    ("kwargs", "names"),
    [
        ({"batch_correction": "scanorama"}, ["scanorama"]),
        ({"batch_correction": "harmony"}, ["harmonypy"]),
        ({"batch_correction": "scvi"}, ["scvi"]),
        ({"batch_correction": "scvi+harmony"}, ["harmonypy", "scvi"]),
        ({"doublet_method": "solo"}, ["scvi"]),
        ({"scfair": True}, ["scfair"]),
        ({"denoise": True, "raw": "raw.h5"}, ["ambidose"]),
    ],
)
def test_missing_optional_package_prints_install_and_keeps_counts(
    monkeypatch, capsys, kwargs, names
):
    import sys

    prepare_module = sys.modules["annotuneagent.prepare"]
    real = prepare_module.find_spec

    def fake(name, package=None):
        if name in names:
            return None
        return real(name, package)

    monkeypatch.setattr(prepare_module, "find_spec", fake)
    adata = _counts()
    adata.obs["sample"] = ["a"] * 40 + ["b"] * 40
    before = adata.X.copy()
    out = prepare(adata, species="human", batch_key="sample", **kwargs)
    assert out is adata
    assert np.array_equal(adata.X, before)
    text = capsys.readouterr().out
    for name in names:
        assert f"{name} is not installed. Install it with:" in text


def test_prepare_rejects_duplicate_gene_names():
    adata = _counts()
    adata.var_names = ["Gene0"] * adata.n_vars
    try:
        prepare(adata, species="mouse")
    except ValueError as exc:
        assert "not unique" in str(exc)
    else:
        raise AssertionError("duplicate var_names were accepted")


def test_elbow_picks_a_bend():
    scores = np.array([10, 6, 3, 2.6, 2.2, 1.8, 1.4])
    assert elbow_count(scores) == 3


def test_elbow_follows_the_shoulder_after_the_steep_drop():
    steep = np.array([0.101, 0.064, 0.061, 0.034, 0.018, 0.016, 0.011])
    shoulder = np.linspace(0.010, 0.0036, 12)
    flat = np.linspace(0.0035, 0.0024, 30)
    scores = np.concatenate([steep, shoulder, flat])
    chosen = elbow_count(scores)
    assert chosen > 7
    assert chosen < len(scores) - 5


def test_shifted_logarithm_scales_to_the_median_library_size():
    from scipy import sparse

    out = prepare(_counts(), species="mouse", n_pcs=5, norm="shifted")
    counts = sparse.csr_matrix(out.layers["counts"])
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scale = float(np.median(totals)) / totals
    expected = sparse.diags(scale).dot(counts).tocsr()
    expected.data = np.log1p(expected.data)
    got = sparse.csr_matrix(out.X)
    assert np.allclose(got.toarray(), expected.toarray())
    assert out.uns["log1p"]["base"] is None
    assert out.uns["annotune_prep"]["norm"] == "shifted"


def test_unknown_norm_is_rejected():
    with pytest.raises(ValueError, match="norm"):
        prepare(_counts(), species="mouse", norm="cpm")


def test_denoise_without_raw_is_refused():
    with pytest.raises(ValueError, match=RAW_REQUIRED):
        prepare(_counts(), species="mouse", denoise=True)


def test_denoise_fits_each_sample_on_its_own_raw_matrix(tmp_path, monkeypatch):
    pytest.importorskip("ambidose")
    import ambidose

    seen = []

    def fake(adata, raw=None, sample_key=None):
        seen.append(
            (
                adata.n_obs,
                tuple(sorted(adata.obs["sample"].astype(str).unique())),
                sample_key,
                str(raw),
                adata.obs_names[0],
            )
        )
        return adata.copy()

    monkeypatch.setattr(ambidose, "denoise", fake)
    monkeypatch.setattr(ambidose, "analysis_ready", lambda adata, **kwargs: adata)
    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "raw_feature_bc_matrix.h5").write_bytes(b"")
    adata = _two_batches()
    adata.obs_names = [f"bc{i}-1-a" if i < 40 else f"bc{i}-1-b" for i in range(80)]
    out = _denoise(adata, raw=tmp_path, sample_key="sample")
    assert [row[:4] for row in seen] == [
        (40, ("a",), None, str(tmp_path / "a" / "raw_feature_bc_matrix.h5")),
        (40, ("b",), None, str(tmp_path / "b" / "raw_feature_bc_matrix.h5")),
    ]
    assert seen[0][4] == "bc0-1"
    assert seen[1][4] == "bc40-1"
    assert out.obs_names[0] == "bc0-1-a"
    assert out.obs_names[40] == "bc40-1-b"
    assert list(out.obs["sample"].astype(str).unique()) == ["a", "b"]


def test_default_resolution_does_not_call_scfair():
    adata = AnnData(np.zeros((4, 4)))
    value, source = annotation_resolution(adata, None)
    assert value == 0.4
    assert source == "default"
    kept, source = annotation_resolution(adata, 0.8)
    assert kept == 0.8
    assert source == "user"


def test_scfair_resolution_is_half_the_suggestion():
    pytest.importorskip("scfair")
    from scfair.pp._diagnosis import resolve_cluster_resolution

    adata = AnnData(np.zeros((4, 4)))
    value, source = annotation_resolution(adata, None, scfair=True)
    assert source == "scfair/2"
    assert value == pytest.approx(float(resolve_cluster_resolution(adata)["resolution"]) / 2)


def test_prepare_clusters_integer_counts_without_denoise():
    adata = prepare(_counts(), species="mouse", n_pcs=5)
    assert "leiden" in adata.obs.columns
    assert adata.obs["leiden"].nunique() >= 1
    assert adata.uns["log1p"]["base"] is None
    prep = adata.uns["annotune_prep"]
    assert prep["denoise"] is False
    assert prep["scfair"] is False
    assert prep["resolution"] == 0.4
    assert prep["resolution_source"] == "default"
    assert prep["n_pcs"] == 5
    assert prep["n_top_genes"] == 39
    assert prep["norm"] == "log1p"
    assert "X_umap" in adata.obsm
    assert prep["n_cells_out"] == prep["n_cells_in"] - prep["n_cells_qc"]
    assert prep["qc"]["max_pct_mt_percent"] is None
    assert prep["n_genes_out"] == prep["n_genes_in"] - prep["n_genes_unobserved"]
    assert prep["batch_correction"] == "none"
    assert prep["doublets"] == {"enabled": False, "method": None, "n_removed": 0, "batches": []}
    assert "use_rep" not in adata.uns["neighbors"]["params"]


def test_doublet_report_counts_each_batch(monkeypatch):
    import scanpy as sc

    seen = []

    def fake(adata, **kwargs):
        assert kwargs["random_state"] == 0
        assert kwargs["expected_doublet_rate"] == pytest.approx(
            _expected_doublet_rate(adata.n_obs, "10k")
        )
        names = adata.obs["sample"].astype(str).unique()
        assert len(names) == 1
        name = str(names[0])
        seen.append((name, adata.n_obs))
        predicted = np.full(adata.n_obs, name == "b")
        adata.obs["predicted_doublet"] = predicted
        adata.obs["doublet_score"] = predicted.astype(float)
        adata.uns["scrublet"] = {"threshold": 0.4 if name == "b" else 0.2}

    monkeypatch.setattr(sc.pp, "scrublet", fake)
    out = prepare(
        _two_batches(),
        species="mouse",
        batch_key="sample",
        n_pcs=5,
        doublet_method="scrublet",
    )
    doublets = out.uns["annotune_prep"]["doublets"]
    assert doublets["enabled"] is True
    assert doublets["method"] == "scrublet"
    assert doublets["chip"] == "10k"
    assert doublets["n_removed"] == 40
    rate_40 = _expected_doublet_rate(40, "10k")
    assert doublets["batches"] == [
        {"batch": "a", "n_cells": 40, "n_removed": 0, "threshold": 0.2, "expected_rate": rate_40},
        {"batch": "b", "n_cells": 40, "n_removed": 40, "threshold": 0.4, "expected_rate": rate_40},
    ]
    assert out.n_obs == 40
    text = _format_filter_report(out.uns["annotune_prep"])
    assert "Scrublet removed 40" in text
    assert "b: 40 cells, 40 removed, threshold 0.4" in text
    bare = prepare(_two_batches(), species="mouse", n_pcs=5, doublet_method="scrublet")
    assert bare.uns["annotune_prep"]["sample_key"] == "sample"
    assert bare.n_obs == 40
    assert seen == [("a", 40), ("b", 40), ("a", 40), ("b", 40)]


def _two_batches():
    adata = _counts()
    adata.obs["sample"] = ["a"] * 40 + ["b"] * 40
    return adata


def test_unknown_batch_correction_is_rejected():
    with pytest.raises(ValueError, match="batch_correction"):
        prepare(_counts(), species="mouse", batch_correction="unknown")


def test_solo_and_scvi_clustering_are_independent(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    import pandas as pd

    calls = []

    class FakeSCVI:
        @staticmethod
        def setup_anndata(adata, layer=None, batch_key=None):
            calls.append((layer, batch_key, adata.n_obs))

        def __init__(self, adata, **kwargs):
            self.adata = adata
            self.n_latent = kwargs.get("n_latent", 10)

        def train(self, **kwargs):
            assert kwargs == {"early_stopping": True}

        def get_latent_representation(self):
            return np.random.default_rng(0).normal(size=(self.adata.n_obs, self.n_latent))

        def save(self, path, overwrite):
            assert overwrite is True
            (tmp_path / "saved.txt").write_text(path)

    class FakeSOLO:
        @classmethod
        def from_scvi_model(cls, vae):
            return cls(vae.adata)

        def __init__(self, adata):
            self.adata = adata

        def train(self, **kwargs):
            assert kwargs == {"early_stopping": True}

        def predict(self, soft=True):
            scores = np.linspace(0, 1, self.adata.n_obs)
            if soft:
                return pd.DataFrame({"doublet": scores}, index=self.adata.obs_names)
            return pd.Series(
                ["singlet"] * (self.adata.n_obs - 2) + ["doublet"] * 2,
                index=self.adata.obs_names,
            )

    fake = SimpleNamespace(
        settings=SimpleNamespace(seed=None),
        model=SimpleNamespace(SCVI=FakeSCVI),
        external=SimpleNamespace(SOLO=FakeSOLO),
    )
    monkeypatch.setitem(sys.modules, "scvi", fake)
    solo_out = prepare(
        _two_batches(),
        species="mouse",
        batch_key="sample",
        n_pcs=5,
        doublet_method="solo",
        solo_expected_rates={"a": 0.1, "b": 0.2},
    )
    assert solo_out.uns["annotune_prep"]["doublets"]["method"] == "solo"
    assert solo_out.uns["annotune_prep"]["n_cells_doublet"] == 12
    assert [x["predicted_rate"] for x in solo_out.uns["annotune_prep"]["doublets"]["batches"]] == [
        0.05,
        0.05,
    ]
    assert [x["removed_rate"] for x in solo_out.uns["annotune_prep"]["doublets"]["batches"]] == [
        0.1,
        0.2,
    ]
    assert solo_out.n_obs == 68
    assert "X_pca" in solo_out.obsm and "X_scVI" not in solo_out.obsm
    assert "solo_score" in solo_out.obs and not solo_out.obs["is_doublet"].any()
    assert [n for layer, _key, n in calls if layer is None] == [40, 40]
    automatic = _drop_solo_doublets(_two_batches(), "sample", None)
    assert automatic["chip"] == "10k"
    assert automatic["n_removed"] == 4
    assert [x["expected_rate"] for x in automatic["batches"]] == [
        _expected_doublet_rate(40, "10k"),
        _expected_doublet_rate(40, "10k"),
    ]
    wide = _drop_solo_doublets(_two_batches(), "sample", None, "20k")
    assert wide["chip"] == "20k"
    assert [x["expected_rate"] for x in wide["batches"]] == [
        _expected_doublet_rate(40, "20k"),
        _expected_doublet_rate(40, "20k"),
    ]
    implied = _drop_solo_doublets(_two_batches(), None, None)
    assert [x["batch"] for x in implied["batches"]] == ["a", "b"]
    assert [x["n_cells"] for x in implied["batches"]] == [40, 40]
    assert implied["n_removed"] == 4
    scvi_out = prepare(
        _two_batches(),
        species="mouse",
        batch_key="sample",
        batch_correction="scvi",
        scvi_model_dir=tmp_path / "model",
    )
    assert scvi_out.uns["annotune_prep"]["doublets"]["enabled"] is False
    assert scvi_out.uns["annotune_prep"]["n_latent"] == 30
    assert scvi_out.uns["neighbors"]["params"]["use_rep"] == "X_scVI"
    assert "X_pca" not in scvi_out.obsm
    assert calls[-1] == ("counts", "sample", scvi_out.n_obs)
    assert (tmp_path / "saved.txt").read_text() == str(tmp_path / "model")
    assert fake.settings.seed == 0


def test_scvi_harmony_uses_the_scvi_latent(monkeypatch):
    import sys
    from types import SimpleNamespace

    pytest.importorskip("harmonypy")
    import harmonypy

    class FakeSCVI:
        @staticmethod
        def setup_anndata(adata, layer=None, batch_key=None):
            return None

        def __init__(self, adata, **kwargs):
            self.adata = adata
            self.n_latent = kwargs.get("n_latent", 10)

        def train(self, **kwargs):
            assert kwargs == {"early_stopping": True}

        def get_latent_representation(self):
            return np.random.default_rng(0).normal(size=(self.adata.n_obs, self.n_latent))

        def save(self, path, overwrite):
            return None

    fake = SimpleNamespace(
        settings=SimpleNamespace(seed=None),
        model=SimpleNamespace(SCVI=FakeSCVI),
        external=SimpleNamespace(SOLO=object),
    )
    monkeypatch.setitem(sys.modules, "scvi", fake)

    def run_harmony(data_mat, meta, key, **kwargs):
        assert key == "sample"
        assert kwargs["random_state"] == 0
        assert kwargs["max_iter_harmony"] == 20
        assert data_mat.shape == (meta.shape[0], 10)

        class Out:
            Z_corr = np.asarray(data_mat, dtype=np.float32)

        return Out()

    monkeypatch.setattr(harmonypy, "run_harmony", run_harmony)
    both = prepare(
        _two_batches(),
        species="mouse",
        batch_key="sample",
        batch_correction="scvi+harmony",
        n_pcs=10,
    )
    assert both.uns["annotune_prep"]["batch_correction"] == "scvi+harmony"
    assert both.uns["annotune_prep"]["n_latent"] == 10
    assert both.uns["neighbors"]["params"]["use_rep"] == "X_scVI_harmony"
    assert both.obsm["X_scVI_harmony"].shape == (both.n_obs, 10)
    assert "X_pca" not in both.obsm


def test_expected_doublet_rate_follows_the_chip():
    assert _expected_doublet_rate(10_000, "10k") == pytest.approx(0.08)
    assert _expected_doublet_rate(10_000, "20k") == pytest.approx(0.04)
    assert _expected_doublet_rate(20_000, "20k") == pytest.approx(0.08)
    assert _expected_doublet_rate(8_695, "10k") == pytest.approx(0.008 * 8695 / 1000)
    with pytest.raises(ValueError, match="doublet_chip"):
        prepare(_counts(), species="mouse", doublet_chip="30k")


def test_stuck_scrublet_threshold_moves_to_the_chip_rate(monkeypatch):
    import scanpy as sc

    n = 1000
    observed = np.linspace(0.05, 0.4, n)

    def fake(adata, **kwargs):
        assert kwargs["expected_doublet_rate"] == pytest.approx(_expected_doublet_rate(n, "10k"))
        assert kwargs["random_state"] == 0
        adata.obs["doublet_score"] = observed
        adata.obs["predicted_doublet"] = np.zeros(n, dtype=bool)
        adata.uns["scrublet"] = {
            "threshold": 0.62,
            "doublet_scores_sim": np.concatenate([np.full(n, 0.2), np.full(n, 0.8)]),
        }

    monkeypatch.setattr(sc.pp, "scrublet", fake)
    adata = AnnData(X=np.ones((n, 4), dtype=np.float64))
    adata.obs_names = [f"c{i}" for i in range(n)]
    adata.var_names = [f"g{i}" for i in range(4)]
    report = _drop_doublets(adata, None, "10k")
    assert report["n_removed"] == 4
    assert report["batches"][0]["threshold"] == pytest.approx(observed[995])
    assert adata.n_obs == n - 4
    kept = _align_scrublet_threshold(
        observed,
        np.concatenate([np.full(n, 0.2), np.full(n, 0.8)]),
        _expected_doublet_rate(n, "10k"),
        observed[995],
    )
    assert kept == pytest.approx(observed[995])


def test_solo_rate_requires_solo_method():
    with pytest.raises(ValueError, match="solo_expected_rates"):
        prepare(_two_batches(), species="mouse", batch_key="sample", solo_expected_rates={"a": 0.1})


def test_solo_rates_must_cover_every_sample():
    with pytest.raises(ValueError, match="missing=\\['b'\\]"):
        prepare(
            _two_batches(),
            species="mouse",
            batch_key="sample",
            doublet_method="solo",
            solo_expected_rates={"a": 0.1},
        )


def test_batch_correction_needs_two_batches():
    with pytest.raises(ValueError, match="batch_key"):
        prepare(_counts(), species="mouse", batch_correction="harmony")
    one = _counts()
    one.obs["sample"] = "only"
    with pytest.raises(ValueError, match="two batches"):
        prepare(one, species="mouse", batch_key="sample", n_pcs=5, batch_correction="harmony")
    single = _counts()
    single.obs["sample"] = "only"
    with pytest.raises(ValueError, match="two batches"):
        prepare(single, species="mouse", batch_key="sample", batch_correction="scvi+harmony")


def test_harmony_stores_cells_by_components():
    import pandas as pd

    harmonypy = pytest.importorskip("harmonypy")
    cells, pcs = 40, 5
    embedding = np.random.default_rng(0).normal(size=(cells, pcs))
    obs = pd.DataFrame({"sample": ["a"] * 20 + ["b"] * 20})
    corrected = harmonypy.run_harmony(
        embedding, obs, "sample", nclust=2, random_state=0, verbose=False
    )
    assert corrected.Z_corr.shape == (cells, pcs)


def test_harmony_output_is_oriented_to_cells_by_components():
    from annotuneagent.prepare import _cells_by_features

    flipped = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert _cells_by_features(flipped, n_cells=4, n_features=3).shape == (4, 3)
    assert _cells_by_features(flipped.T, n_cells=4, n_features=3).shape == (4, 3)


def test_harmony_and_scanorama_feed_the_neighbor_graph(monkeypatch):
    pytest.importorskip("harmonypy")
    pytest.importorskip("scanorama")
    import harmonypy
    import scanpy as sc

    def run_harmony(data_mat, meta, key, **kwargs):
        assert key == "sample"
        assert data_mat.shape == (meta.shape[0], 5)
        n_cells = meta.shape[0]

        class Out:
            Z_corr = np.ones((n_cells, 5), dtype=np.float32)

        return Out()

    def scanorama(adata, key, **kwargs):
        assert key == "sample"
        adata.obsm["X_scanorama"] = np.asarray(adata.obsm["X_pca"])

    monkeypatch.setattr(harmonypy, "run_harmony", run_harmony)
    monkeypatch.setattr(sc.external.pp, "scanorama_integrate", scanorama)
    harmony_out = prepare(
        _two_batches(), species="mouse", batch_key="sample", n_pcs=5, batch_correction="harmony"
    )
    assert harmony_out.obsm["X_pca_harmony"].shape == (harmony_out.n_obs, 5)
    assert harmony_out.uns["annotune_prep"]["batch_correction"] == "harmony"
    assert harmony_out.uns["neighbors"]["params"]["use_rep"] == "X_pca_harmony"
    scan_out = prepare(
        _two_batches(),
        species="mouse",
        batch_key="sample",
        n_pcs=5,
        batch_correction="scanorama",
    )
    assert scan_out.uns["neighbors"]["params"]["use_rep"] == "X_scanorama"


def test_combat_leaves_log1p_expression_in_x():
    out = prepare(
        _two_batches(), species="mouse", batch_key="sample", n_pcs=5, batch_correction="combat"
    )
    assert out.uns["neighbors"]["params"]["use_rep"] == "X_pca_combat"
    expected = out.copy()
    expected.X = expected.layers["counts"].copy()
    sc = pytest.importorskip("scanpy")
    sc.pp.normalize_total(expected, target_sum=1e4)
    sc.pp.log1p(expected)
    got = out.X.toarray() if hasattr(out.X, "toarray") else np.asarray(out.X)
    ref = expected.X.toarray() if hasattr(expected.X, "toarray") else np.asarray(expected.X)
    assert np.allclose(got, ref)


def _varied_counts():
    rng = np.random.default_rng(2)
    x = rng.integers(1, 6, size=(70, 16)).astype(np.float64)
    x[0] = 0
    x[0, 1] = 1
    x[1] = x[1] + 400
    x[2, :] = 0
    x[2, 0] = 50
    adata = AnnData(X=x)
    adata.var_names = ["mt-A", "mt-B", *[f"Gene{i}" for i in range(14)]]
    adata.obs_names = [f"c{i}" for i in range(adata.n_obs)]
    adata.obs["batch"] = ["a"] * 35 + ["b"] * 35
    return adata


def test_show_prints_qc_rules_and_cell_counts(capsys, monkeypatch):
    import matplotlib.pyplot as plt
    import scanpy as sc

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    for name in ("highest_expr_genes", "violin", "scatter", "highly_variable_genes", "umap"):
        monkeypatch.setattr(sc.pl, name, lambda *args, **kwargs: None)
    out = prepare(_counts(), species="mouse", n_pcs=5, show=True)
    prep = out.uns["annotune_prep"]
    assert prep["n_cells_out"] == (
        prep["n_cells_after_denoise"] - prep["n_cells_qc"] - prep["n_cells_doublet"]
    )
    assert sum(batch["n_removed"] for batch in prep["qc"]["batches"]) == prep["n_cells_qc"]
    assert prep["qc"]["batches"][0]["batch"] == "all"
    assert prep["qc"]["mt_prefixes"] == ["mt-", "Mt-"]
    report = _format_filter_report(prep)
    assert "MAD" not in report
    assert "doublets" in report and "off" in report
    assert "mitochondrial percent cap: off" in report
    rules = {rule["metric"]: rule for rule in prep["qc"]["batches"][0]["rules"]}
    total = rules["log1p_total_counts"]
    low = int(round(_on_count_scale("log1p_total_counts", total["low"])))
    high = int(round(_on_count_scale("log1p_total_counts", total["high"])))
    assert f"total counts: {low:,} to {high:,}" in report
    genes = rules["log1p_n_genes_by_counts"]
    glow = int(round(_on_count_scale("log1p_n_genes_by_counts", genes["low"])))
    ghigh = int(round(_on_count_scale("log1p_n_genes_by_counts", genes["high"])))
    assert f"genes per cell: {glow:,} to {ghigh:,}" in report
    mito = rules["logit_mt_fraction"]
    mpct = _on_count_scale("logit_mt_fraction", mito["high"])
    assert f"mitochondrial percent: {mpct:.1f}".rstrip("0").rstrip(".") in report
    assert report in capsys.readouterr().out


def test_qc_report_matches_direct_mad_counts():
    import scanpy as sc
    from scipy.special import logit

    probe = _varied_counts()
    qc = _filter_qc(probe.copy(), "mouse", "batch", 5.0, 90)
    assert qc["sample_key"] == "batch"
    assert qc["max_pct_mt_percent"] == 90
    assert [batch["batch"] for batch in qc["batches"]] == ["a", "b"]
    names = probe.var_names.astype(str)
    probe.var["mt"] = names.str.startswith(("mt-", "Mt-"))
    probe.var["ribo"] = names.str.startswith(("Rps", "Rpl", "RPS", "RPL"))
    probe.var["hb"] = names.str.contains(r"^(?:HB|Hb)[ABDEGMQZabdegmqz]", regex=True)
    sc.pp.calculate_qc_metrics(
        probe, qc_vars=["mt", "ribo", "hb"], percent_top=None, log1p=True, inplace=True
    )
    fraction = (probe.obs["total_counts_mt"] + 1) / (probe.obs["total_counts"] + 2)
    probe.obs["logit_mt_fraction"] = logit(np.clip(fraction, 1e-6, 1 - 1e-6))
    for batch in qc["batches"]:
        mask = probe.obs["batch"].astype(str).to_numpy() == batch["batch"]
        pct = probe.obs["pct_counts_mt"].to_numpy()[mask]
        assert batch["n_max_pct_mt"] == int((pct > 90).sum())
        union = pct > 90
        assert [rule["metric"] for rule in batch["rules"]] == [
            "log1p_total_counts",
            "log1p_n_genes_by_counts",
            "logit_mt_fraction",
        ]
        for rule in batch["rules"]:
            values = probe.obs[rule["metric"]].to_numpy()[mask]
            med = float(np.median(values))
            mad = float(stats.median_abs_deviation(values))
            scale = 3.0 if rule["metric"] == "logit_mt_fraction" else 5.0
            assert rule["median"] == pytest.approx(med)
            assert rule["mad"] == pytest.approx(mad)
            if not np.isfinite(mad) or mad == 0:
                assert rule["n_removed"] == 0
                continue
            if rule["side"] == "both":
                hit = (values < med - scale * mad) | (values > med + scale * mad)
                assert rule["low"] == pytest.approx(med - scale * mad)
            else:
                hit = values > med + scale * mad
                assert rule["low"] is None
                hit = hit & (pct > 8)
            assert rule["high"] == pytest.approx(med + scale * mad)
            assert rule["n_removed"] == int(hit.sum())
            union = union | hit
        assert batch["n_removed"] == int(union.sum())
    text = _format_filter_report(
        {
            "denoise": False,
            "doublet_method": "none",
            "n_cells_in": 70,
            "n_cells_denoise": 0,
            "n_cells_after_denoise": 70,
            "n_cells_qc": qc["n_removed"],
            "n_cells_doublet": 0,
            "n_cells_out": 70 - qc["n_removed"],
            "n_genes_in": 16,
            "n_genes_denoise": 0,
            "n_genes_unobserved": 0,
            "n_genes_out": 16,
            "doublet_thresholds": [],
            "qc": qc,
        }
    )
    assert "mitochondrial percent cap: 90" in text
    assert "\na:" in text
    assert "\nb:" in text
    assert "above 90" in text


def test_qc_thresholds_are_fit_inside_each_sample():
    values_a = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 30], dtype=float)
    values_b = np.full(10, 30.0)
    counts = np.expm1(np.concatenate([values_a, values_b]))[:, None]
    adata = AnnData(X=counts)
    adata.var_names = ["Gene0"]
    adata.obs_names = [f"c{i}" for i in range(adata.n_obs)]
    adata.obs["sample"] = ["a"] * 10 + ["b"] * 10
    qc = _filter_qc(adata, "mouse", "sample", 5.0, None)
    assert qc["n_removed"] == 1
    assert "c9" not in set(adata.obs_names)
    assert set(adata.obs["sample"].astype(str)) == {"a", "b"}


@pytest.mark.parametrize("value", [0, -5, 100.5, 1000])
def test_max_pct_mt_is_a_percent_from_zero_to_a_hundred(value):
    adata = AnnData(X=np.ones((10, 4)))
    adata.var_names = ["A", "B", "C", "D"]
    with pytest.raises(ValueError, match="percent from 0 to 100"):
        prepare(adata, species="human", max_pct_mt=value)


def test_prepare_with_scfair_picks_genes_and_halves_the_resolution():
    pytest.importorskip("scfair")
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0, (300, 300)).astype(float)
    counts[:150, :20] += rng.poisson(5, (150, 20))
    adata = AnnData(counts)
    adata.var_names = [f"G{i}" for i in range(300)]
    adata.obs_names = [f"c{i}" for i in range(300)]
    result = prepare(adata, species="human", scfair=True)
    prep = result.uns["annotune_prep"]
    assert prep["scfair"] is True
    assert prep["resolution_source"] == "scfair/2"
    assert prep["n_top_genes"] > 0


def test_max_pct_mt_below_one_warns_that_it_is_a_percent():
    with pytest.warns(UserWarning, match="not a fraction"):
        prepare(_counts(), species="mouse", n_pcs=5, max_pct_mt=0.5)
