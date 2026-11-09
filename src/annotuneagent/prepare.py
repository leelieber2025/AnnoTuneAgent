# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Count matrix to a Leiden clustering for annotation.

AmbiDose ambient subtraction is off unless ``denoise`` is set, and then it
needs the unfiltered droplet matrix. Harmony, Scanorama, scVI, scFair, and
AmbiDose are optional: a missing package prints an install line and prepare
returns the same matrix. Highly variable genes are Scanpy
``seurat_v3`` and the Leiden resolution is 0.4. ``scfair=True`` uses scFair
for both. Batch correction defaults to none. Harmony, ComBat, Scanorama and
scVI change the neighbor representation. ``X`` stays log1p.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy import sparse, stats
from scipy.special import expit, logit

from .evidence import require_unique_var_names

_SPECIES = ("human", "mouse")
_MT = {"human": ("MT-",), "mouse": ("mt-", "Mt-")}
_RESOLUTION_SCALE = 0.5
_DEFAULT_RESOLUTION = 0.4
_DEFAULT_N_TOP_GENES = 2000
_MAD_SCALE = 5.0
_MT_NMADS = 3.0
_MT_MIN_PCT = 8.0
_TOP_GENES = 20
_PCA_CAP = 50
_RIBO = {"human": ("RPS", "RPL"), "mouse": ("Rps", "Rpl", "RPS", "RPL")}
_HB = {
    "human": r"^HB[ABDEGMQZ]",
    "mouse": r"^(?:HB|Hb)[ABDEGMQZabdegmqz]",
}
_BATCH_CORRECTION = ("none", "harmony", "combat", "scanorama", "scvi", "scvi+harmony")
_NORMS = ("log1p", "shifted")
_DOUBLET_METHODS = ("none", "scrublet", "solo")
_DOUBLET_CHIPS = ("10k", "20k")
# Selected only. None of these are installed with the package.
_OPTIONAL = {
    "harmonypy": "pip install harmonypy",
    "scanorama": "pip install scanorama",
    "scvi": "pip install 'annotuneagent[scvi]'",
    "scfair": "pip install 'annotuneagent[scfair]'",
    "ambidose": "pip install ambidose",
}
# 10x CG000422, Next GEM Single Cell 3' v3.1. Standard chip: 0.8% multiplets per
# 1,000 recovered cells. HT chip: 0.4% per 1,000. Either chip is about 8% at its
# rated load (10,000 or 20,000 recovered cells).
_MULTIPLETS_PER_CELL = {"10k": 0.008 / 1000, "20k": 0.004 / 1000}
RAW_REQUIRED = "No unfiltered droplet matrix. Pass raw, or leave AmbiDose off."


@contextmanager
def _timed(name: str) -> Iterator[dict[str, str]]:
    """Print a step when it starts and its elapsed seconds when it finishes."""
    print(f"{name} ...", flush=True)
    started = time.perf_counter()
    note: dict[str, str] = {}
    try:
        yield note
    finally:
        detail = note.get("detail", "")
        suffix = f"  {detail}" if detail else ""
        print(f"{name}  {time.perf_counter() - started:.1f}s{suffix}", flush=True)


def prepare(
    adata: AnnData,
    *,
    species: str,
    batch_key: str | None = None,
    batch_correction: str = "none",
    doublet_method: str = "none",
    doublet_chip: str = "10k",
    solo_expected_rates: dict[str, float] | None = None,
    scvi_model_dir: str | Path | None = None,
    denoise: bool = False,
    raw: str | Path | AnnData | dict[str, str | Path | AnnData] | None = None,
    resolution: float | None = None,
    n_pcs: int | None = None,
    n_top_genes: int | None = None,
    scfair: bool = False,
    max_pct_mt: float | None = None,
    cluster_key: str = "leiden",
    n_neighbors: int = 15,
    nmads: float = _MAD_SCALE,
    norm: str = "log1p",
    show: bool = False,
) -> AnnData:
    """Filter, select genes, and cluster. The result is natural log1p in ``X``.

    ``norm="log1p"`` scales each cell to 10,000 counts and applies ``log1p``.
    ``norm="shifted"`` is the shifted logarithm: scale each cell to the median
    library size, then ``log1p``. Counts stay in ``layers["counts"]``.

    Highly variable genes are Scanpy ``seurat_v3`` and the Leiden resolution is
    0.4. ``scfair=True`` uses scFair for both. ``batch_correction`` is ``none``,
    ``harmony``, ``combat``, ``scanorama``, ``scvi``, or ``scvi+harmony``.
    ``scvi+harmony`` trains scVI, then runs Harmony on that latent space.
    Correction changes the neighbor graph only, and ``X`` stays log1p.
    ``max_pct_mt`` is a percent, so 10 keeps
    cells with at most 10% mitochondrial counts. QC thresholds, Scrublet, and
    SOLO are fit separately inside each sample. ``batch_key`` names that column.
    When it is omitted and ``obs['sample']`` exists, that column is used.
    ``doublet_method`` selects no removal, Scrublet, or SOLO. Both use the
    filtered matrix already in ``adata``: unnormalized UMI counts of called
    cells, not ``raw_feature_bc_matrix``. ``doublet_chip`` is ``10k`` (0.8%
    multiplets per 1,000 cells in that sample) or ``20k`` (0.4% per 1,000).
    That rate is Scrublet's prior. SOLO uses it too, unless
    ``solo_expected_rates`` gives one fraction per sample. Denoising fits
    one raw matrix per sample. With several samples, ``raw`` is the parent
    folder (``base/<sample>/raw_feature_bc_matrix.h5``, or the mtx directory
    when no h5 is present), a dict keyed by the sample name in ``batch_key``,
    or one AnnData that has that same column. One sample may be a single h5,
    mtx directory, or AnnData.
    Each step prints when it starts and how many
    seconds it took. ``show=True`` then prints the QC rules and draws the QC
    figures, the variance elbow, and the UMAP.
    """
    if install_hint(
        batch_correction=batch_correction,
        doublet_method=doublet_method,
        scfair=scfair,
        denoise=denoise,
    ):
        return adata
    if species not in _SPECIES:
        raise ValueError(f"species must be human or mouse, got {species!r}")
    require_unique_var_names(adata)
    if max_pct_mt is not None and not 0 < max_pct_mt <= 100:
        raise ValueError(f"max_pct_mt is a percent from 0 to 100, got {max_pct_mt!r}")
    if max_pct_mt is not None and max_pct_mt < 1:
        warnings.warn(
            f"max_pct_mt={max_pct_mt} is {max_pct_mt}%, not a fraction; "
            f"{max_pct_mt * 100:g} means {max_pct_mt * 100:g}%",
            UserWarning,
            stacklevel=2,
        )
    correction = batch_correction.strip().casefold()
    if correction not in _BATCH_CORRECTION:
        allowed = ", ".join(_BATCH_CORRECTION)
        raise ValueError(f"batch_correction must be one of {allowed}, got {batch_correction!r}")
    if doublet_method not in _DOUBLET_METHODS:
        raise ValueError(f"doublet_method must be one of {', '.join(_DOUBLET_METHODS)}")
    chip = doublet_chip.strip().casefold()
    if chip not in _DOUBLET_CHIPS:
        allowed = ", ".join(_DOUBLET_CHIPS)
        raise ValueError(f"doublet_chip must be one of {allowed}, got {doublet_chip!r}")
    transform = norm.strip().casefold()
    if transform not in _NORMS:
        raise ValueError(f"norm must be one of {', '.join(_NORMS)}, got {norm!r}")
    sample_key = _resolve_sample_key(adata, batch_key)
    if scvi_model_dir is not None and correction not in ("scvi", "scvi+harmony"):
        raise ValueError("scvi_model_dir requires batch_correction='scvi' or 'scvi+harmony'")
    if solo_expected_rates is not None and doublet_method != "solo":
        raise ValueError("solo_expected_rates requires doublet_method='solo'")
    if solo_expected_rates is not None:
        samples = {"all"} if sample_key is None else set(adata.obs[sample_key].astype(str))
        if set(solo_expected_rates) != samples:
            raise ValueError(
                "solo_expected_rates must specify every sample exactly once; "
                f"missing={sorted(samples - set(solo_expected_rates))}, "
                f"unknown={sorted(set(solo_expected_rates) - samples)}"
            )
        if any(
            not isinstance(rate, (int, float)) or not np.isfinite(rate) or not 0 <= rate < 1
            for rate in solo_expected_rates.values()
        ):
            raise ValueError("SOLO expected rates must be finite fractions in [0, 1)")
    if (
        correction in ("scvi", "scvi+harmony") or doublet_method == "solo"
    ) and not adata.obs_names.is_unique:
        raise ValueError("scVI/SOLO requires unique cell names in adata.obs_names")
    n_in = int(adata.n_obs)
    n_genes_in = int(adata.n_vars)
    started = time.perf_counter()
    if denoise:
        if raw is None:
            raise ValueError(RAW_REQUIRED)
        n_samples = 1 if sample_key is None else int(adata.obs[sample_key].nunique())
        with _timed("prepare  denoise") as note:
            adata = _denoise(adata, raw=raw, sample_key=sample_key)
            grouped = f"{n_samples} samples, " if n_samples > 1 else ""
            note["detail"] = f"{grouped}{adata.n_obs} cells"
    elif not _integer_counts(adata.X):
        raise ValueError("prepare expects raw integer counts in adata.X")
    if (correction in ("scvi", "scvi+harmony") or doublet_method == "solo") and not _integer_counts(
        adata.X
    ):
        raise ValueError("scVI and SOLO require integer raw counts after denoising")
    n_after_denoise = int(adata.n_obs)
    n_genes_after_denoise = int(adata.n_vars)
    with _timed("prepare  drop genes with no counts") as note:
        n_unobserved = _drop_unobserved_genes(adata)
        note["detail"] = f"{n_unobserved} genes"
    with _timed("prepare  QC") as note:
        qc = _filter_qc(adata, species, sample_key, nmads, max_pct_mt)
        grouped = f", per {sample_key}" if sample_key else ""
        note["detail"] = f"{qc['n_removed']} cells removed, {adata.n_obs} remain{grouped}"
    n_qc = int(qc["n_removed"])
    if doublet_method != "none":
        with _timed("prepare  doublets") as note:
            doublets = (
                _drop_solo_doublets(adata, sample_key, solo_expected_rates, chip)
                if doublet_method == "solo"
                else _drop_doublets(adata, sample_key, chip)
            )
            note["detail"] = (
                f"{doublets['n_removed']} removed in {len(doublets['batches'])} samples"
            )
    else:
        doublets = {"enabled": False, "method": None, "n_removed": 0, "batches": []}
    n_doublet = int(doublets["n_removed"])
    if adata.n_obs < 3 or adata.n_vars < 2:
        raise ValueError("too few cells or genes remain after filtering")
    with _timed("prepare  highly variable genes") as note:
        n_genes = _select_genes(adata, n_top_genes, sample_key, scfair=scfair)
        note["detail"] = f"{n_genes} genes"
    with _timed(f"prepare  normalize {transform}") as note:
        adata.layers["counts"] = adata.X.copy()
        _normalize(adata, transform)
        note["detail"] = transform
    latent_dims = None
    if correction in ("scvi", "scvi+harmony"):
        if correction == "scvi+harmony":
            _require_batches(adata, batch_key)
        with _timed("prepare  scVI") as note:
            latent_dims = 30 if n_pcs is None else int(n_pcs)
            if latent_dims < 1:
                raise ValueError("n_pcs must be at least 1 for scVI latent dimensions")
            _fit_scvi(adata, batch_key, latent_dims, scvi_model_dir)
            note["detail"] = f"n_latent {latent_dims}"
        chosen_pcs = 0
        use_rep = "X_scVI"
        if correction == "scvi+harmony":
            with _timed("prepare  harmony") as note:
                adata.obsm["X_scVI_harmony"] = _run_harmony(
                    adata.obsm["X_scVI"], adata.obs, batch_key
                )
                use_rep = "X_scVI_harmony"
                note["detail"] = use_rep
    else:
        with _timed("prepare  PCA") as note:
            chosen_pcs = _fit_pcs(adata, n_pcs)
            note["detail"] = f"n_pcs {chosen_pcs}"
    if correction not in ("none", "scvi", "scvi+harmony"):
        with _timed(f"prepare  {correction}") as note:
            use_rep = _integrate(adata, correction, batch_key, chosen_pcs)
            note["detail"] = use_rep
    elif correction == "none":
        use_rep = _integrate(adata, correction, batch_key, chosen_pcs)
    neighbors = min(n_neighbors, int(adata.n_obs) - 1)
    n_graph = (
        int(adata.obsm[use_rep].shape[1])
        if correction in ("scvi", "scvi+harmony")
        else min(chosen_pcs, int(adata.obsm[use_rep].shape[1]))
    )
    with _timed("prepare  neighbors"):
        sc.pp.neighbors(
            adata,
            n_neighbors=neighbors,
            n_pcs=n_graph,
            use_rep=None if use_rep == "X_pca" else use_rep,
            random_state=0,
        )
    used_resolution, resolution_source = annotation_resolution(adata, resolution, scfair=scfair)
    with _timed("prepare  Leiden") as note:
        sc.tl.leiden(
            adata,
            resolution=used_resolution,
            key_added=cluster_key,
            flavor="igraph",
            n_iterations=2,
            directed=False,
            random_state=0,
        )
        note["detail"] = (
            f"resolution {used_resolution:g} ({resolution_source}), "
            f"{adata.obs[cluster_key].nunique()} clusters"
        )
    with _timed("prepare  UMAP"):
        sc.tl.umap(adata, random_state=0)
    print(
        f"prepare  done  {time.perf_counter() - started:.1f}s  "
        f"{adata.n_obs} cells, {adata.obs[cluster_key].nunique()} clusters",
        flush=True,
    )
    adata.uns["annotune_prep"] = {
        "species": species,
        "batch_key": batch_key,
        "sample_key": sample_key,
        "batch_correction": correction,
        "doublet_method": doublet_method,
        "denoise": bool(denoise),
        "raw": None if raw is None or isinstance(raw, AnnData) else str(raw),
        "norm": transform,
        "scfair": bool(scfair),
        "n_top_genes": int(n_genes),
        "n_pcs": int(chosen_pcs),
        "n_latent": None if latent_dims is None else int(latent_dims),
        "scvi_model_dir": None if scvi_model_dir is None else str(scvi_model_dir),
        "n_neighbors": int(neighbors),
        "resolution": float(used_resolution),
        "resolution_source": resolution_source,
        "cluster_key": cluster_key,
        "nmads": float(nmads),
        "max_pct_mt": None if max_pct_mt is None else float(max_pct_mt),
        "n_cells_in": n_in,
        "n_cells_denoise": n_in - n_after_denoise,
        "n_cells_after_denoise": n_after_denoise,
        "n_cells_qc": n_qc,
        "n_cells_doublet": int(n_doublet),
        "n_cells_out": int(adata.n_obs),
        "n_genes_in": n_genes_in,
        "n_genes_denoise": n_genes_in - n_genes_after_denoise,
        "n_genes_unobserved": n_unobserved,
        "n_genes_out": int(adata.n_vars),
        "qc": qc,
        "doublets": doublets,
    }
    if show:
        _show_prepared(adata, cluster_key)
    return adata


def _show_prepared(adata: AnnData, cluster_key: str) -> None:
    import matplotlib.pyplot as plt

    print(_format_filter_report(adata.uns["annotune_prep"]))
    sc.pl.highest_expr_genes(adata, n_top=20, layer="counts", show=True)
    _qc_violin(adata)
    sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts", show=True)
    sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt", show=True)
    sc.pl.highly_variable_genes(adata, show=True)
    n_pcs = int(adata.uns["annotune_prep"]["n_pcs"])
    if n_pcs:
        ratio = adata.uns["pca"]["variance_ratio"]
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.plot(range(1, len(ratio) + 1), ratio, color="0.2")
        ax.axvline(n_pcs, color="C0", ls="--", label=f"n_pcs = {n_pcs}")
        ax.set_xlabel("principal component")
        ax.set_ylabel("variance ratio")
        ax.legend(frameon=False)
        fig.tight_layout()
        plt.show()
    print(adata.obs[cluster_key].value_counts().sort_index())
    sc.pl.umap(adata, color=cluster_key, legend_loc="on data", title=cluster_key, show=True)


def _qc_violin(adata: AnnData) -> None:
    """Gene count, UMI count, and mitochondrial percent, split by sample when present."""
    import matplotlib.pyplot as plt

    keys = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    sample_key = adata.uns["annotune_prep"]["qc"].get("sample_key")
    if not sample_key or sample_key not in adata.obs.columns:
        sc.pl.violin(adata, keys, jitter=0.4, multi_panel=True, show=True)
        return
    axes = sc.pl.violin(
        adata,
        keys,
        groupby=sample_key,
        jitter=0.4,
        multi_panel=True,
        rotation=45,
        show=False,
    )
    if axes is None:
        return
    plt.tight_layout()
    plt.show()


def annotation_resolution(
    adata: AnnData,
    resolution: float | None,
    *,
    scfair: bool = False,
) -> tuple[float, str]:
    """User resolution is kept. The default is 0.4 unless ``scfair`` is set."""
    if resolution is not None:
        return float(resolution), "user"
    if not scfair:
        return _DEFAULT_RESOLUTION, "default"
    from scfair.pp._diagnosis import resolve_cluster_resolution

    suggested = float(resolve_cluster_resolution(adata)["resolution"])
    return suggested * _RESOLUTION_SCALE, "scfair/2"


def _chord_gap(values: np.ndarray, minimum: int) -> tuple[int, float]:
    """1-based index of the largest vertical gap under the endpoint chord."""
    n = int(values.size)
    if n <= minimum:
        return n, 0.0
    x = np.linspace(0.0, 1.0, n)
    chord = values[0] + (values[-1] - values[0]) * x
    start = max(minimum - 1, 1)
    gap = chord - values
    index = start + int(np.argmax(gap[start:]))
    return index + 1, float(gap[index])


def elbow_count(scores: np.ndarray, *, minimum: int = 2) -> int:
    """How many leading scores to keep.

    The largest gap under the chord is the steep drop. Components after that
    drop can still fall. When that remainder sits well below its own chord,
    keep up to the bend of the remainder. A straight remainder keeps the
    first drop.
    """
    values = np.asarray(scores, dtype=float)
    n = int(values.size)
    if n <= minimum:
        return n
    first, _gap = _chord_gap(values, minimum)
    if first >= n - 1:
        return first
    tail = values[first - 1 :]
    local, second_gap = _chord_gap(tail, 2)
    span = float(tail[0] - tail[-1])
    if span <= 0 or second_gap / span < 0.2:
        return first
    return first - 1 + local


def _require_batches(adata: AnnData, batch_key: str | None) -> None:
    if batch_key is None or batch_key not in adata.obs.columns:
        raise ValueError("batch correction needs batch_key")
    if adata.obs[batch_key].nunique(dropna=False) < 2:
        raise ValueError("batch correction needs at least two batches")


def _integrate(adata: AnnData, method: str, batch_key: str | None, n_pcs: int) -> str:
    """Representation for the neighbor graph. Expression in ``X`` is unchanged."""
    if method == "none":
        return "X_pca"
    _require_batches(adata, batch_key)
    if method == "harmony":
        _harmony(adata, batch_key, n_pcs)
        return "X_pca_harmony"
    if method == "scanorama":
        _scanorama(adata, batch_key, n_pcs)
        return "X_scanorama"
    _combat_pca(adata, batch_key, n_pcs)
    return "X_pca_combat"


def _harmony(adata: AnnData, batch_key: str, n_pcs: int) -> None:
    """Correct the selected principal components. ``obsm`` stays cells by components."""
    pcs = np.asarray(adata.obsm["X_pca"], dtype=np.float64)[:, :n_pcs]
    if pcs.shape != (adata.n_obs, n_pcs):
        raise ValueError(f"X_pca shape {pcs.shape} is not {(adata.n_obs, n_pcs)}")
    adata.obsm["X_pca_harmony"] = _run_harmony(pcs, adata.obs, batch_key)


def _run_harmony(matrix, obs, batch_key: str) -> np.ndarray:
    """Return Harmony output as cells by features.

    harmonypy 0.2 returns ``Z_corr`` as cells by components. The matrix is
    transposed only when it comes back as components by cells.
    """
    import harmonypy

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Harmony input shape {values.shape} is not a matrix")
    if not np.isfinite(values).all():
        raise ValueError("Harmony input contains NaN or Inf")
    n_cells, n_features = values.shape
    corrected = _cells_by_features(
        harmonypy.run_harmony(
            values,
            obs,
            batch_key,
            max_iter_harmony=20,
            random_state=0,
        ).Z_corr,
        n_cells,
        n_features,
    )
    if not np.isfinite(corrected).all():
        raise ValueError("Harmony corrected matrix contains NaN or Inf")
    return np.ascontiguousarray(corrected, dtype=np.float32)


def _cells_by_features(matrix, n_cells: int, n_features: int) -> np.ndarray:
    if hasattr(matrix, "detach"):
        matrix = matrix.detach().cpu().numpy()
    values = np.asarray(matrix)
    if values.shape == (n_cells, n_features):
        return values
    if values.shape == (n_features, n_cells):
        return values.T
    raise ValueError(
        f"Harmony output shape {values.shape} is neither "
        f"{(n_cells, n_features)} nor {(n_features, n_cells)}"
    )


def _scanorama(adata: AnnData, batch_key: str, n_pcs: int) -> None:
    labels = adata.obs[batch_key].astype(str).to_numpy()
    order = np.argsort(labels, kind="mergesort")
    obs = adata.obs.iloc[order][[batch_key]].copy()
    small = AnnData(X=np.zeros((order.size, 1), dtype=np.float32), obs=obs)
    small.obsm["X_pca"] = np.asarray(adata.obsm["X_pca"])[order, :n_pcs]
    sc.external.pp.scanorama_integrate(small, batch_key)
    back = np.empty_like(small.obsm["X_scanorama"])
    back[order] = np.asarray(small.obsm["X_scanorama"])
    adata.obsm["X_scanorama"] = back


def _combat_pca(adata: AnnData, batch_key: str, n_pcs: int) -> None:
    keep = adata.var["highly_variable"].to_numpy()
    if int(keep.sum()) <= n_pcs:
        raise ValueError("not enough highly variable genes for ComBat")
    sub = adata[:, keep].copy()
    sc.pp.combat(sub, key=batch_key)
    sc.pp.pca(sub, n_comps=n_pcs, svd_solver="arpack", random_state=0)
    adata.obsm["X_pca_combat"] = np.asarray(sub.obsm["X_pca"])


def _denoise(
    adata: AnnData,
    *,
    raw: str | Path | AnnData | dict[str, str | Path | AnnData],
    sample_key: str | None,
) -> AnnData:
    """Estimate and subtract ambient RNA inside one sample at a time."""
    labels = _sample_labels(adata, sample_key)
    names = list(dict.fromkeys(labels.tolist()))
    pieces = []
    for name in names:
        idx = np.flatnonzero(labels == name)
        sub = adata[idx].copy()
        if sample_key is not None and sub.obs[sample_key].astype(str).nunique() != 1:
            raise RuntimeError(f"denoise subset {name} contains more than one sample")
        raw_one = _raw_for_sample(raw, str(name), n_samples=len(names), sample_key=sample_key)
        if len(names) > 1:
            print(f"prepare  denoise  {name}  {sub.n_obs} cells  {raw_one}", flush=True)
        pieces.append(_denoise_one(sub, raw_one, sample=None if sample_key is None else str(name)))
    if len(pieces) == 1:
        return pieces[0]
    merged = ad.concat(pieces, join="outer", merge="same")
    keep = [cell for cell in adata.obs_names if cell in merged.obs_names]
    merged = merged[keep].copy()
    if sample_key is not None:
        merged.obs[sample_key] = merged.obs[sample_key].astype("category")
    return merged


def _denoise_one(adata: AnnData, raw: str | Path | AnnData, *, sample: str | None) -> AnnData:
    import ambidose

    names = [str(cell) for cell in adata.obs_names]
    stripped, restore = _strip_sample_suffix(names, sample)
    if restore:
        adata.obs_names = stripped
    denoised = ambidose.denoise(adata, raw=raw, sample_key=None)
    out = ambidose.analysis_ready(denoised)
    if restore:
        out.obs_names = [f"{cell}-{sample}" for cell in out.obs_names.astype(str)]
    return out


def _strip_sample_suffix(names: list[str], sample: str | None) -> tuple[list[str], bool]:
    """Drop the ``read_10x`` sample suffix so barcodes match the raw matrix."""
    if sample is None:
        return names, False
    suffix = f"-{sample}"
    if not all(name.endswith(suffix) for name in names):
        return names, False
    if sample.isdigit() and all(name.count("-") == 1 for name in names):
        return names, False
    stripped = [name[: -len(suffix)] for name in names]
    if any(name == "" for name in stripped) or len(set(stripped)) != len(stripped):
        return names, False
    return stripped, True


def _raw_for_sample(
    raw: str | Path | AnnData | dict[str, str | Path | AnnData],
    sample: str,
    *,
    n_samples: int,
    sample_key: str | None,
) -> str | Path | AnnData:
    if isinstance(raw, dict):
        try:
            return raw[sample]
        except KeyError:
            raise KeyError(f"raw has no matrix for sample {sample!r}") from None
    if isinstance(raw, AnnData):
        if sample_key is not None and sample_key in raw.obs.columns:
            chosen = raw[raw.obs[sample_key].astype(str).to_numpy() == sample].copy()
            if chosen.n_obs == 0:
                raise ValueError(f"raw has no droplets for sample {sample!r}")
            return chosen
        if n_samples == 1:
            return raw
        raise ValueError("raw AnnData has no sample column; pass each sample its own raw matrix")
    path = Path(raw)
    if n_samples == 1 and (path.is_file() or _contains_raw(path)):
        return path
    folder = path / sample
    if folder.is_dir():
        return _find_count_matrix(folder, "raw_feature_bc_matrix.h5", "raw_feature_bc_matrix")
    raise FileNotFoundError(f"no raw matrix for sample {sample!r} under {path}")


def _contains_raw(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        _find_count_matrix(path, "raw_feature_bc_matrix.h5", "raw_feature_bc_matrix")
    except FileNotFoundError:
        return False
    return True


def _find_count_matrix(folder: Path, h5_name: str, mtx_name: str) -> Path:
    found: list[Path] = []
    for current, dirnames, filenames in os.walk(folder):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        here = Path(current)
        if h5_name in filenames:
            found.append(here / h5_name)
        mtx = here / mtx_name
        if mtx_name in dirnames and any(
            (mtx / name).is_file() for name in ("matrix.mtx", "matrix.mtx.gz")
        ):
            found.append(mtx)
    if not found:
        raise FileNotFoundError(f"{folder} has no {h5_name} or {mtx_name}")
    return min(found, key=lambda path: (len(path.parts), path.suffix != ".h5", str(path)))


def _normalize(adata: AnnData, transform: str) -> None:
    """Library-size normalization followed by a natural log1p.

    ``shifted`` follows the sc-best-practices shifted logarithm: the target
    is the median library size. ``log1p`` keeps the fixed target of 10,000.
    """
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()
    if transform == "shifted":
        totals = np.asarray(adata.X.sum(axis=1)).ravel()
        if totals.size == 0 or not np.isfinite(totals).all() or float(np.median(totals)) <= 0:
            raise ValueError("shifted logarithm needs a positive median library size")
        sc.pp.normalize_total(adata, target_sum=None)
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


def _integer_counts(matrix) -> bool:
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    if values.size == 0:
        return True
    values = np.asarray(values)
    return bool(
        np.isfinite(values).all()
        and (values >= 0).all()
        and np.max(np.abs(values - np.rint(values))) < 1e-6
    )


def _drop_unobserved_genes(adata: AnnData) -> int:
    totals = np.asarray(adata.X.sum(axis=0)).ravel()
    keep = totals > 0
    n_removed = int((~keep).sum())
    if n_removed:
        adata._inplace_subset_var(keep)
    return n_removed


def _resolve_sample_key(adata: AnnData, batch_key: str | None) -> str | None:
    """Column that splits QC and doublet detection into samples."""
    if batch_key is not None:
        if batch_key not in adata.obs.columns:
            raise KeyError(f"batch_key={batch_key!r} not in adata.obs")
        return batch_key
    if "sample" in adata.obs.columns:
        return "sample"
    return None


def _sample_labels(adata: AnnData, sample_key: str | None) -> np.ndarray:
    if sample_key is None:
        return np.full(adata.n_obs, "all", dtype=object)
    return adata.obs[sample_key].astype(str).to_numpy()


def _filter_qc(
    adata: AnnData,
    species: str,
    sample_key: str | None,
    nmads: float,
    max_pct_mt: float | None,
) -> dict:
    names = adata.var_names.astype(str)
    adata.var["mt"] = names.str.startswith(_MT[species])
    adata.var["ribo"] = names.str.startswith(_RIBO[species])
    adata.var["hb"] = names.str.contains(_HB[species], regex=True)
    top_n = _TOP_GENES if adata.n_vars >= _TOP_GENES else None
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        percent_top=None if top_n is None else [top_n],
        log1p=True,
        inplace=True,
    )
    total_mt = adata.obs["total_counts_mt"].to_numpy(dtype=float)
    total = adata.obs["total_counts"].to_numpy(dtype=float)
    fraction = np.clip((total_mt + 1.0) / (total + 2.0), 1e-6, 1.0 - 1e-6)
    adata.obs["logit_mt_fraction"] = logit(fraction)
    labels = _sample_labels(adata, sample_key)
    outlier = np.zeros(adata.n_obs, dtype=bool)
    percent = None if max_pct_mt is None else float(max_pct_mt)
    metrics = [
        ("log1p_total_counts", float(nmads), False),
        ("log1p_n_genes_by_counts", float(nmads), False),
    ]
    if top_n is not None:
        metrics.append((f"pct_counts_in_top_{top_n}_genes", float(nmads), False))
    reported: list[dict] = []
    for batch in np.unique(labels):
        mask = labels == batch
        idx = np.flatnonzero(mask)
        rules = []
        batch_outlier = np.zeros(idx.shape[0], dtype=bool)
        for metric, scale, upper in metrics:
            rule, hit = _mad_rule(adata.obs[metric].to_numpy()[mask], scale, upper=upper)
            rule["metric"] = metric
            rule["side"] = "upper" if upper else "both"
            rule["nmads"] = float(scale)
            rules.append(rule)
            batch_outlier |= hit
        logit_values = adata.obs["logit_mt_fraction"].to_numpy()[mask]
        mt_rule, mt_hit = _mad_rule(logit_values, _MT_NMADS, upper=True)
        pct = adata.obs["pct_counts_mt"].to_numpy()[mask]
        above_min = pct > _MT_MIN_PCT
        mt_removed = mt_hit & above_min
        mt_rule["metric"] = "logit_mt_fraction"
        mt_rule["side"] = "upper"
        mt_rule["nmads"] = float(_MT_NMADS)
        mt_rule["min_pct"] = float(_MT_MIN_PCT)
        mt_rule["n_mad"] = int(mt_hit.sum())
        mt_rule["n_above_min_pct"] = int(above_min.sum())
        mt_rule["n_removed"] = int(mt_removed.sum())
        rules.append(mt_rule)
        batch_outlier |= mt_removed
        n_cap = 0 if percent is None else int((pct > percent).sum())
        if percent is not None:
            batch_outlier |= pct > percent
        outlier[idx] |= batch_outlier
        reported.append(
            {
                "batch": str(batch),
                "n_cells": int(mask.sum()),
                "rules": rules,
                "n_max_pct_mt": n_cap,
                "n_removed": int(batch_outlier.sum()),
            }
        )
    n_removed = int(outlier.sum())
    if n_removed:
        adata._inplace_subset_obs(~outlier)
    return {
        "nmads": float(nmads),
        "mt_nmads": float(_MT_NMADS),
        "mt_min_pct": float(_MT_MIN_PCT),
        "sample_key": sample_key,
        "batch_key": sample_key,
        "mt_prefixes": list(_MT[species]),
        "max_pct_mt": None if max_pct_mt is None else float(max_pct_mt),
        "max_pct_mt_percent": None if percent is None else float(percent),
        "batches": reported,
        "n_removed": n_removed,
    }


def _mad_rule(values: np.ndarray, nmads: float, *, upper: bool) -> tuple[dict, np.ndarray]:
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    mad = float(stats.median_abs_deviation(values))
    rule = {
        "median": med,
        "mad": mad,
        "low": None,
        "high": None,
        "n_low": 0,
        "n_high": 0,
        "n_removed": 0,
    }
    if not np.isfinite(mad) or mad == 0:
        return rule, np.zeros(values.shape[0], dtype=bool)
    if upper:
        high = med + nmads * mad
        hit = values > high
        rule["high"] = float(high)
        rule["n_high"] = int(hit.sum())
    else:
        low = med - nmads * mad
        high = med + nmads * mad
        below = values < low
        above = values > high
        hit = below | above
        rule["low"] = float(low)
        rule["high"] = float(high)
        rule["n_low"] = int(below.sum())
        rule["n_high"] = int(above.sum())
    rule["n_removed"] = int(hit.sum())
    return rule, hit


def _expected_doublet_rate(n_cells: int, chip: str) -> float:
    """10x multiplet rate for cells recovered on a 10k or 20k chip."""
    rate = _MULTIPLETS_PER_CELL[chip] * int(n_cells)
    if rate >= 1:
        raise ValueError(f"{n_cells} cells on a {chip} chip implies a multiplet rate above 1")
    return float(rate)


def _overall_doublet_rate(obs: np.ndarray, sim: np.ndarray, threshold: float) -> float:
    """Scrublet's estimated doublet rate at one score cutoff.

    Observed calls divided by the fraction of simulated doublets above the
    cutoff. ``call_doublets`` compares this with ``expected_doublet_rate``.
    """
    detectable = int(np.count_nonzero(sim > threshold))
    if detectable == 0 or len(obs) == 0:
        return float("inf")
    detected = int(np.count_nonzero(obs > threshold)) / len(obs)
    return detected / (detectable / len(sim))


def _align_scrublet_threshold(
    obs: np.ndarray,
    sim: np.ndarray,
    expected: float,
    automatic: float | None,
) -> float:
    """Move a cutoff that calls too few doublets toward the 10x rate.

    A cutoff that already reaches the expected rate stays. The replacement
    minimizes the gap between Scrublet's overall rate and ``expected``.
    """
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    if automatic is not None and np.isfinite(automatic):
        current = _overall_doublet_rate(obs, sim, float(automatic))
        if np.isfinite(current) and current >= 0.75 * expected:
            return float(automatic)
    scores = np.unique(np.concatenate([obs, sim]))
    scores = scores[np.isfinite(scores)]
    best_t = None
    best_key = None
    for value in scores:
        rate = _overall_doublet_rate(obs, sim, float(value))
        if not np.isfinite(rate):
            continue
        key = (abs(rate - expected), -float(value))
        if best_key is None or key < best_key:
            best_key = key
            best_t = float(value)
    if best_t is not None:
        return best_t
    n_call = int(round(expected * len(obs)))
    if n_call <= 0:
        return float(np.nanmax(obs)) + 1.0
    cutoff = np.partition(obs, len(obs) - n_call)[len(obs) - n_call]
    return float(np.nextafter(np.float64(cutoff), -np.inf))


def _scrublet_sim_scores(adata: AnnData) -> np.ndarray | None:
    info = adata.uns.get("scrublet")
    if not isinstance(info, dict):
        return None
    sim = info.get("doublet_scores_sim")
    if sim is None:
        return None
    values = np.asarray(sim, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).any():
        return None
    return values


def _drop_doublets(adata: AnnData, sample_key: str | None, chip: str) -> dict:
    """Scrublet fits and calls doublets inside one sample at a time."""
    labels = _sample_labels(adata, sample_key)
    predicted = np.zeros(adata.n_obs, dtype=bool)
    scores = np.full(adata.n_obs, np.nan)
    position = {cell: index for index, cell in enumerate(adata.obs_names)}
    batches = []
    for name in np.unique(labels):
        idx = np.flatnonzero(labels == name)
        sub = adata[idx].copy()
        if sample_key is not None and sub.obs[sample_key].astype(str).nunique() != 1:
            raise RuntimeError(f"Scrublet subset {name} contains more than one sample")
        expected = _expected_doublet_rate(sub.n_obs, chip)
        sc.pp.scrublet(sub, expected_doublet_rate=expected, random_state=0)
        if "predicted_doublet" not in sub.obs.columns:
            raise RuntimeError(f"scrublet did not write predicted_doublet for {name}")
        called = sub.obs["predicted_doublet"].fillna(False).to_numpy().astype(bool)
        score = sub.obs["doublet_score"].to_numpy(dtype=float)
        found = _scrublet_thresholds(sub)
        threshold = found.get(str(name), found.get("all"))
        sim = _scrublet_sim_scores(sub)
        if sim is not None:
            current = (
                0.0 if threshold is None else _overall_doublet_rate(score, sim, float(threshold))
            )
            if not (np.isfinite(current) and current >= 0.75 * expected):
                threshold = _align_scrublet_threshold(score, sim, expected, threshold)
                called = score > threshold
        for cell, flag, value in zip(sub.obs_names, called, score, strict=True):
            at = position[cell]
            predicted[at] = bool(flag)
            scores[at] = float(value)
        batches.append(
            {
                "batch": str(name),
                "n_cells": int(idx.size),
                "n_removed": int(predicted[idx].sum()),
                "threshold": None if threshold is None else float(threshold),
                "expected_rate": expected,
            }
        )
    adata.obs["doublet_score"] = scores
    adata.obs["predicted_doublet"] = predicted
    n_removed = int(predicted.sum())
    if n_removed:
        adata._inplace_subset_obs(~predicted)
    return {
        "enabled": True,
        "method": "scrublet",
        "chip": chip,
        "n_removed": n_removed,
        "batches": batches,
    }


def install_hint(
    *,
    batch_correction: str = "none",
    doublet_method: str = "none",
    scfair: bool = False,
    denoise: bool = False,
) -> bool:
    """Print an install line for each selected package that is missing.

    Returns True when at least one is missing. The caller leaves the matrix
    unchanged.
    """
    correction = batch_correction.strip().casefold()
    needed: list[str] = []
    if correction in ("harmony", "scvi+harmony"):
        needed.append("harmonypy")
    if correction == "scanorama":
        needed.append("scanorama")
    if correction in ("scvi", "scvi+harmony") or doublet_method == "solo":
        needed.append("scvi")
    if scfair:
        needed.append("scfair")
    if denoise:
        needed.append("ambidose")
    missing = [name for name in needed if find_spec(name) is None]
    for name in missing:
        print(f"{name} is not installed. Install it with: {_OPTIONAL[name]}", flush=True)
    return bool(missing)


def _scvi_module():
    import scvi

    return scvi


def _drop_solo_doublets(
    adata: AnnData,
    batch_key: str | None,
    expected_rates: dict[str, float] | None,
    chip: str = "10k",
) -> dict:
    """SOLO fits and calls doublets inside one sample at a time."""
    sample_key = _resolve_sample_key(adata, batch_key)
    labels = _sample_labels(adata, sample_key)
    names = set(np.unique(labels))
    if expected_rates is not None and names - set(expected_rates):
        raise ValueError(
            f"SOLO expected rates missing batches: {sorted(names - set(expected_rates))}"
        )
    scvi = _scvi_module()
    scores = np.empty(adata.n_obs, dtype=float)
    predictions = np.empty(adata.n_obs, dtype=object)
    removed = np.zeros(adata.n_obs, dtype=bool)
    batches = []
    for name in np.unique(labels):
        idx = np.flatnonzero(labels == name)
        sub = adata[idx].copy()
        if sample_key is not None and sub.obs[sample_key].astype(str).nunique() != 1:
            raise RuntimeError(f"SOLO subset {name} contains more than one sample")
        scvi.model.SCVI.setup_anndata(sub)
        vae = scvi.model.SCVI(sub, gene_likelihood="nb")
        vae.train(early_stopping=True)
        solo = scvi.external.SOLO.from_scvi_model(vae)
        solo.train(early_stopping=True)
        soft = solo.predict(soft=True)
        hard = solo.predict(soft=False)
        batch_scores = soft.loc[sub.obs_names, "doublet"].to_numpy(dtype=float)
        batch_labels = np.asarray(hard.loc[sub.obs_names]).reshape(-1).astype(str)
        if not np.isfinite(batch_scores).all():
            raise ValueError(f"SOLO returned non-finite doublet probabilities for {name}")
        scores[idx] = batch_scores
        predictions[idx] = batch_labels
        batch_removed = batch_labels == "doublet"
        predicted_rate = float(batch_removed.mean())
        if expected_rates is None:
            rate = _expected_doublet_rate(int(idx.size), chip)
        else:
            rate = float(expected_rates[str(name)])
        threshold = None
        if rate is not None:
            n_top = int(np.ceil(rate * idx.size))
            if n_top:
                ranked = np.argsort(batch_scores, kind="stable")[-n_top:]
                batch_removed[ranked] = True
                threshold = float(batch_scores[ranked].min())
        removed[idx] = batch_removed
        batches.append(
            {
                "batch": str(name),
                "n_cells": int(idx.size),
                "n_removed": int(batch_removed.sum()),
                "predicted_rate": predicted_rate,
                "removed_rate": float(batch_removed.mean()),
                "expected_rate": rate,
                "threshold": threshold,
            }
        )
    adata.obs["solo_score"] = scores
    adata.obs["solo_prediction"] = predictions
    adata.obs["is_doublet"] = removed
    n_removed = int(removed.sum())
    if n_removed:
        adata._inplace_subset_obs(~removed)
    return {
        "enabled": True,
        "method": "solo",
        "chip": chip if expected_rates is None else None,
        "n_removed": n_removed,
        "batches": batches,
    }


def _fit_scvi(
    adata: AnnData, batch_key: str | None, n_latent: int, model_dir: str | Path | None
) -> None:
    scvi = _scvi_module()
    scvi.settings.seed = 0
    genes = adata.var["highly_variable"].to_numpy(dtype=bool)
    train = adata[:, genes].copy()
    scvi.model.SCVI.setup_anndata(train, layer="counts", batch_key=batch_key)
    model = scvi.model.SCVI(train, n_layers=2, n_latent=n_latent, gene_likelihood="nb")
    model.train(early_stopping=True)
    adata.obsm["X_scVI"] = np.asarray(model.get_latent_representation())
    if model_dir is not None:
        model.save(str(model_dir), overwrite=True)


def _scrublet_thresholds(adata: AnnData) -> dict[str, float]:
    info = adata.uns.get("scrublet")
    if not isinstance(info, dict):
        return {}
    found: dict[str, float] = {}
    batches = info.get("batches")
    if isinstance(batches, dict):
        for name, batch in batches.items():
            threshold = batch.get("threshold") if isinstance(batch, dict) else None
            if threshold is not None:
                found[str(name)] = float(threshold)
        return found
    threshold = info.get("threshold")
    if threshold is not None:
        found["all"] = float(threshold)
    return found


def _format_filter_report(prep: dict) -> str:
    """QC rules and the cell and gene counts at each filtering step."""
    after_denoise = int(prep["n_cells_after_denoise"])
    after_qc = after_denoise - int(prep["n_cells_qc"])
    lines = [
        _step("batch correction", prep.get("batch_correction", "none")),
        "",
        "Cells",
        _step("input", _count(prep["n_cells_in"])),
    ]
    if prep["denoise"]:
        lines.append(
            _step("AmbiDose", f"-{_count(prep['n_cells_denoise'])} -> {_count(after_denoise)}")
        )
    else:
        lines.append(_step("AmbiDose", "off"))
    lines.append(_step("QC", f"-{_count(prep['n_cells_qc'])} -> {_count(after_qc)}"))
    lines.extend(_qc_threshold_lines(prep))
    doublets = prep.get("doublets") or {
        "enabled": False,
        "n_removed": prep.get("n_cells_doublet", 0),
    }
    if doublets.get("enabled"):
        method = "SOLO" if doublets.get("method") == "solo" else "Scrublet"
        lines.append(
            _step(method, f"-{_count(doublets['n_removed'])} -> {_count(prep['n_cells_out'])}")
        )
        lines.extend(f"    {line}" for line in _doublet_batch_lines(doublets))
    else:
        lines.append(_step("doublets", "off"))
    lines.extend(
        [
            _step("output", _count(prep["n_cells_out"])),
            "",
            "Genes",
            _step("input", _count(prep["n_genes_in"])),
        ]
    )
    if prep["denoise"] and prep["n_genes_denoise"]:
        after_gene_denoise = int(prep["n_genes_in"]) - int(prep["n_genes_denoise"])
        lines.append(
            _step("AmbiDose", f"-{_count(prep['n_genes_denoise'])} -> {_count(after_gene_denoise)}")
        )
    lines.extend(
        [
            _step(
                "zero total",
                f"-{_count(prep['n_genes_unobserved'])} -> {_count(prep['n_genes_out'])}",
            ),
            "",
            f"Mitochondrial genes: {', '.join(prep['qc']['mt_prefixes'])}.",
            "Ribosomal and hemoglobin fractions are recorded, not filtered.",
        ]
    )
    cap = prep["qc"]["max_pct_mt_percent"]
    if cap is None:
        lines.append("  mitochondrial percent cap: off")
    else:
        lines.append(f"  mitochondrial percent cap: {_fmt(cap)}")
    lines.extend(["", format_doublets(doublet_summary(prep))])
    return "\n".join(lines)


def _qc_threshold_lines(prep: dict) -> list[str]:
    """Per-sample cutoffs in the units a methods section can quote."""
    qc = prep.get("qc") or {}
    lines = []
    cap = qc.get("max_pct_mt_percent")
    for batch in qc.get("batches") or []:
        lines.append(
            f"{batch['batch']}: {_count(batch['n_cells'])} cells, "
            f"{_count(batch['n_removed'])} removed"
        )
        lines.extend(f"  {_format_rule(rule)}" for rule in batch["rules"])
        if cap is not None:
            lines.append(
                f"  percent mitochondrial counts above {_fmt(cap)}: "
                f"{_count(batch['n_max_pct_mt'])} removed"
            )
    return lines


def doublet_summary(prep: dict | str | None) -> dict:
    """Doublet removal recorded by ``prepare``. Missing prep means it was not run."""
    if isinstance(prep, str):
        prep = json.loads(prep)
    if not isinstance(prep, dict):
        return {"enabled": None, "method": None, "n_removed": None, "batches": []}
    stored = prep.get("doublets")
    if isinstance(stored, dict):
        return stored
    return {"enabled": None, "method": None, "n_removed": None, "batches": []}


def format_doublets(doublets: dict) -> str:
    """Text block for the doublet step. An unknown step says it was not recorded."""
    if doublets.get("enabled") is None:
        return "Doublets\n  not recorded"
    if not doublets.get("enabled"):
        return "Doublets\n  off"
    method = "SOLO" if doublets.get("method") == "solo" else "Scrublet"
    lines = ["Doublets", f"  {method} removed {_count(doublets['n_removed'])}"]
    lines.extend(_doublet_batch_lines(doublets))
    return "\n".join(lines)


def _doublet_batch_lines(doublets: dict) -> list[str]:
    lines = []
    for batch in doublets.get("batches") or []:
        threshold = batch.get("threshold")
        cutoff = "no threshold" if threshold is None else f"threshold {_fmt(threshold)}"
        expected = batch.get("expected_rate")
        if expected is not None:
            cutoff = f"{cutoff}, expected {100 * float(expected):.1f}%"
        if batch.get("n_cells") is None:
            lines.append(f"{batch['batch']}: {cutoff}")
            continue
        lines.append(
            f"{batch['batch']}: {_count(batch['n_cells'])} cells, "
            f"{_count(batch['n_removed'])} removed, {cutoff}"
        )
    return lines


def _qc_label(metric: str) -> str:
    if metric == "log1p_total_counts":
        return "total counts"
    if metric == "log1p_n_genes_by_counts":
        return "genes per cell"
    if metric.startswith("pct_counts_in_top_") and metric.endswith("_genes"):
        n = metric.removeprefix("pct_counts_in_top_").removesuffix("_genes")
        return f"percent counts in top {n} genes"
    if metric == "logit_mt_fraction":
        return "mitochondrial percent"
    return metric


def _on_count_scale(metric: str, value: float | None) -> float | None:
    """log1p cutoffs become UMI or gene counts. The mito logit becomes a percent."""
    if value is None or not np.isfinite(value):
        return None
    if metric.startswith("log1p_"):
        return float(np.expm1(value))
    if metric == "logit_mt_fraction":
        return float(100.0 * expit(value))
    return float(value)


def _fmt_threshold(metric: str, value: float | None) -> str:
    scaled = _on_count_scale(metric, value)
    if scaled is None:
        return "none"
    if metric.startswith("log1p_"):
        return f"{int(round(scaled)):,}"
    if metric.startswith("pct_") or metric == "logit_mt_fraction":
        text = f"{scaled:.1f}"
        return text[:-2] if text.endswith(".0") else text
    return _fmt(scaled)


def _format_rule(rule: dict) -> str:
    metric = rule["metric"]
    label = _qc_label(metric)
    removed = _count(rule["n_removed"])
    if not np.isfinite(rule["mad"]) or rule["mad"] == 0:
        return f"{label}: no cutoff, {removed} removed"
    if metric == "logit_mt_fraction":
        high = _fmt_threshold(metric, rule["high"])
        floor = _fmt(rule["min_pct"])
        return f"{label}: {high} and {floor}, {removed} removed"
    if rule["side"] == "both":
        low = _fmt_threshold(metric, rule["low"])
        high = _fmt_threshold(metric, rule["high"])
        return (
            f"{label}: {low} to {high}, {removed} removed "
            f"({_count(rule['n_low'])} below, {_count(rule['n_high'])} above)"
        )
    return f"{label}: {_fmt_threshold(metric, rule['high'])}, {removed} removed"


def _step(label: str, detail: str) -> str:
    return f"  {label:<16} {detail}"


def _count(value: int) -> str:
    return f"{int(value):,}"


def _fmt(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value)):,}"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    return f"{value:.4g}"


def _select_genes(
    adata: AnnData,
    n_top_genes: int | None,
    batch_key: str | None,
    *,
    scfair: bool,
) -> int:
    if scfair:
        import scfair as scf

        options = scf.pp.HVGOptions(
            batch_key=batch_key,
            append_budget=None if n_top_genes is None else 0,
        )
        scf.pp.highly_variable_genes(
            adata,
            n_top_genes="auto" if n_top_genes is None else int(n_top_genes),
            options=options,
        )
    else:
        count = _DEFAULT_N_TOP_GENES if n_top_genes is None else int(n_top_genes)
        if count < 1:
            raise ValueError("n_top_genes must be at least 1")
        count = min(count, int(adata.n_vars) - 1)
        if count < 1:
            adata.var["highly_variable"] = True
        else:
            try:
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=count,
                    flavor="seurat_v3",
                    batch_key=batch_key,
                )
            except ValueError:
                if batch_key is None:
                    raise
                # A tiny batch makes the seurat_v3 loess fit singular.
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=count,
                    flavor="seurat_v3",
                    batch_key=None,
                )
    if "highly_variable" not in adata.var.columns or not adata.var["highly_variable"].any():
        adata.var["highly_variable"] = True
    return int(adata.var["highly_variable"].sum())


def _fit_pcs(adata: AnnData, n_pcs: int | None) -> int:
    n_hvg = (
        int(adata.var["highly_variable"].sum())
        if "highly_variable" in adata.var.columns
        else adata.n_vars
    )
    cap = min(_PCA_CAP, int(adata.n_obs) - 1, max(n_hvg, 2) - 1, int(adata.n_vars) - 1)
    if cap < 2:
        raise ValueError("not enough cells or genes for PCA")
    sc.pp.pca(adata, n_comps=cap, mask_var="highly_variable", svd_solver="arpack", random_state=0)
    if n_pcs is not None:
        chosen = int(n_pcs)
        if chosen < 1 or chosen > cap:
            raise ValueError(f"n_pcs must be from 1 to {cap}")
        return chosen
    ratio = np.asarray(adata.uns["pca"]["variance_ratio"], dtype=float)
    return min(elbow_count(ratio), cap)
