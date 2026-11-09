# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Gene-level evidence for one cluster against the other cells."""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numba
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse, stats

# A marker is relatively high inside the cluster. Other cell types may also
# express it. A negative marker counts against a label only when a substantial
# fraction of the cluster expresses it and its level is higher inside
# (log fold-change). Equal inside and outside levels are ambient and do not.
NEGATIVE_PRESENT_PCT = 0.25
NEGATIVE_MIN_LOGFC = 1.0


_LOG1P_MAX = 30.0
# Marker columns are copied once. The cap keeps that copy near 256 MB.
_MARKER_CHUNK_BYTES = 256 * 1024 * 1024
# Per-thread cluster sums stay under this. Above it, one thread owns the sum.
_PARTIAL_SUM_BYTES = 1024 * 1024 * 1024


def _cpus() -> int:
    return max(1, os.cpu_count() or 1)


def _expression_values(adata: AnnData) -> np.ndarray:
    if sparse.issparse(adata.X):
        return np.asarray(adata.X.data)
    return np.asarray(adata.X).ravel()


def _looks_like_log1p(values: np.ndarray) -> bool:
    """True when values are already natural-log1p and not raw counts.

    Raw counts are whole numbers. A missing ``uns['log1p']`` flag does not
    by itself mean the matrix is counts.
    """
    if values.size == 0:
        return True
    peak = float(np.max(values))
    if peak > _LOG1P_MAX:
        return False
    nonzero = values[values > 0]
    if nonzero.size == 0 or peak < 1:
        return True
    whole = np.abs(nonzero - np.rint(nonzero)) < 1e-6
    return float(whole.mean()) <= 0.5


def validate_expression(adata: AnnData) -> None:
    """Require natural-log1p, library-size-normalized expression in ``X``.

    A matrix that passes and has no ``uns['log1p']`` flag gets
    ``uns['log1p'] = {'base': None}``.
    """
    flagged = adata.uns.get("log1p")
    if isinstance(flagged, dict) and flagged.get("base") is not None:
        raise ValueError("adata.X must use natural log1p expression")
    values = _expression_values(adata)
    if values.size and (not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError("adata.X must contain finite, nonnegative log1p expression")
    if flagged is None:
        if not _looks_like_log1p(values):
            raise ValueError("adata.X must be normalized with scanpy.pp.normalize_total and log1p")
        adata.uns["log1p"] = {"base": None}


@dataclass(frozen=True)
class GeneEvidence:
    gene: str
    direction: str
    pct_in: float
    pct_out: float
    mean_in: float
    mean_out: float
    logfc: float
    pvalue: float | None
    expressed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "gene": self.gene,
            "direction": self.direction,
            "pct_in": self.pct_in,
            "pct_out": self.pct_out,
            "mean_in": self.mean_in,
            "mean_out": self.mean_out,
            "logfc": self.logfc,
            "pvalue": self.pvalue,
            "expressed": self.expressed,
        }


def _cluster_labels(adata: AnnData, cluster_key: str) -> np.ndarray:
    """Cluster label of every cell as text. A cell with no cluster is an error."""
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"obs has no column {cluster_key!r}")
    column = adata.obs[cluster_key]
    missing = int(column.isna().sum())
    if missing:
        raise ValueError(
            f"obs[{cluster_key!r}] has {missing} cells with no cluster; assign or remove them first"
        )
    return column.astype(str).to_numpy()


def cluster_ids(adata: AnnData, cluster_key: str) -> list[str]:
    return list(dict.fromkeys(_cluster_labels(adata, cluster_key).tolist()))


def cluster_mask(adata: AnnData, cluster_key: str, cluster: str) -> np.ndarray:
    mask = _cluster_labels(adata, cluster_key) == str(cluster)
    if not mask.any():
        raise ValueError(f"cluster {cluster!r} has no cells")
    if mask.all():
        raise ValueError(f"cluster {cluster!r} contains every cell; there is no contrast")
    return mask


def require_unique_var_names(adata: AnnData) -> None:
    """Stop before marker scoring when two columns share a gene symbol."""
    names = pd.Index(adata.var_names.astype(str))
    if names.is_unique:
        return
    raise ValueError("var_names are not unique; call adata.var_names_make_unique() first")


def evidence_from_stats(
    gene: str,
    direction: str,
    stats_row: tuple[float, float, float, float, float, float],
) -> GeneEvidence:
    pct_in, pct_out, mean_in, mean_out, logfc, pvalue = stats_row
    expressed = (
        negative_marker_conflicts(pct_in, pct_out, logfc) if direction == "negative" else logfc > 0
    )
    return GeneEvidence(
        gene=gene,
        direction=direction,
        pct_in=pct_in,
        pct_out=pct_out,
        mean_in=mean_in,
        mean_out=mean_out,
        logfc=logfc,
        pvalue=pvalue,
        expressed=expressed,
    )


def measure_gene(
    adata: AnnData,
    mask: np.ndarray,
    gene: str,
    *,
    direction: str,
) -> GeneEvidence:
    """One marker against the cells outside ``mask``.

    The number is the same value ``marker_statistics`` reports. Mann-Whitney
    p-values are recorded and do not change the score.
    """
    row = marker_statistics(adata, [np.asarray(mask, dtype=bool)], [gene])[0][gene]
    return evidence_from_stats(gene, direction, row)


def marker_statistics(
    adata: AnnData,
    masks: list[np.ndarray],
    genes: list[str],
) -> list[dict[str, tuple[float, float, float, float, float, float]]]:
    """Percent, linear mean, log fold-change, and Mann-Whitney p for every mask.

    Each value is ``(pct_in, pct_out, mean_in, mean_out, logfc, pvalue)``.
    Threads share ``adata.X``. A process pool would copy the matrix per worker.
    """
    n_obs = adata.n_obs
    n_clusters = len(masks)
    n_genes = len(genes)
    if n_clusters == 0:
        return []
    codes = np.full(n_obs, -1, dtype=np.int32)
    n_in = np.zeros(n_clusters, dtype=np.float64)
    for cluster, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (n_obs,):
            raise ValueError("cluster mask length does not match n_obs")
        count = int(mask.sum())
        if count == 0 or count == n_obs:
            raise ValueError("`x` and `y` must be of nonzero size.")
        codes[mask] = np.int32(cluster)
        n_in[cluster] = count
    empty = [{} for _ in range(n_clusters)]
    if n_genes == 0:
        return empty
    require_unique_var_names(adata)
    found = pd.Index(adata.var_names.astype(str)).get_indexer(genes)
    for gene, position in zip(genes, found, strict=True):
        if position < 0:
            raise KeyError(f"gene {gene!r} is not in var_names")
    indexes = [int(position) for position in found]
    pct_in = np.zeros((n_clusters, n_genes), dtype=np.float64)
    pct_out = np.zeros((n_clusters, n_genes), dtype=np.float64)
    mean_in = np.zeros((n_clusters, n_genes), dtype=np.float64)
    mean_out = np.zeros((n_clusters, n_genes), dtype=np.float64)
    logfc = np.zeros((n_clusters, n_genes), dtype=np.float64)
    pvalues = np.ones((n_clusters, n_genes), dtype=np.float64)
    tie_term = np.zeros(n_genes, dtype=np.float64)
    chunk = max(1, _MARKER_CHUNK_BYTES // (8 * max(n_obs, 1)))
    for offset in range(0, n_genes, chunk):
        stop = min(n_genes, offset + chunk)
        block = _gene_block(adata.X, indexes[offset:stop])
        _score_block(
            block,
            offset,
            codes,
            n_in,
            pct_in,
            pct_out,
            mean_in,
            mean_out,
            logfc,
            pvalues,
            tie_term,
        )
        _exact_small_clusters(block, codes, n_in, pvalues, tie_term, offset)
    rows: list[dict[str, tuple[float, float, float, float, float, float]]] = []
    for cluster in range(n_clusters):
        rows.append(
            {
                gene: (
                    float(pct_in[cluster, index]),
                    float(pct_out[cluster, index]),
                    float(mean_in[cluster, index]),
                    float(mean_out[cluster, index]),
                    float(logfc[cluster, index]),
                    float(pvalues[cluster, index]),
                )
                for index, gene in enumerate(genes)
            }
        )
    return rows


def _gene_block(matrix, indexes: list[int]) -> np.ndarray:
    chosen = np.asarray(indexes, dtype=int)
    if sparse.issparse(matrix):
        dense = np.asarray(matrix[:, chosen].toarray(), dtype=np.float64)
    else:
        dense = np.asarray(matrix[:, chosen], dtype=np.float64)
    if dense.ndim == 1:
        dense = dense.reshape(-1, 1)
    return np.ascontiguousarray(dense.T)


def _score_block(
    block: np.ndarray,
    offset: int,
    codes: np.ndarray,
    n_in: np.ndarray,
    pct_in: np.ndarray,
    pct_out: np.ndarray,
    mean_in: np.ndarray,
    mean_out: np.ndarray,
    logfc: np.ndarray,
    pvalues: np.ndarray,
    tie_term: np.ndarray,
) -> None:
    n_chunk = block.shape[0]

    def call(start: int, stop: int) -> None:
        _marker_chunk(
            block,
            codes,
            n_in,
            pct_in,
            pct_out,
            mean_in,
            mean_out,
            logfc,
            pvalues,
            tie_term,
            offset,
            start,
            stop,
        )

    _run_chunks(call, n_chunk)


def _run_chunks(call, n_items: int) -> None:
    if n_items <= 0:
        return
    workers = min(_cpus(), n_items)
    edges = [int(i * n_items / workers) for i in range(workers + 1)]
    # Compile on this thread, then give every core a slice.
    call(0, 0)
    if workers == 1:
        call(0, n_items)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(call, edges[i], edges[i + 1]) for i in range(workers)]
        for future in futures:
            future.result()


@numba.njit(cache=True, nogil=True)
def _marker_chunk(
    block,
    codes,
    n_in,
    pct_in,
    pct_out,
    mean_in,
    mean_out,
    logfc,
    pvalues,
    tie_term,
    offset,
    start,
    stop,
):
    n_obs = block.shape[1]
    n_clusters = n_in.shape[0]
    for local in range(start, stop):
        column = block[local]
        gene = offset + local
        sum_in = np.zeros(n_clusters, dtype=np.float64)
        count_in = np.zeros(n_clusters, dtype=np.float64)
        total_sum = 0.0
        total_count = 0.0
        first = column[0]
        constant = True
        for index in range(n_obs):
            value = column[index]
            if value != first:
                constant = False
            linear = math.expm1(value)
            total_sum += linear
            expressed = 1.0 if value > 0.0 else 0.0
            total_count += expressed
            cluster = codes[index]
            if cluster >= 0:
                sum_in[cluster] += linear
                count_in[cluster] += expressed
        for cluster in range(n_clusters):
            n1 = n_in[cluster]
            n2 = n_obs - n1
            mean_in[cluster, gene] = sum_in[cluster] / n1
            mean_out[cluster, gene] = (total_sum - sum_in[cluster]) / n2
            pct_in[cluster, gene] = count_in[cluster] / n1
            pct_out[cluster, gene] = (total_count - count_in[cluster]) / n2
            logfc[cluster, gene] = math.log2(
                (mean_in[cluster, gene] + 1.0) / (mean_out[cluster, gene] + 1.0)
            )
            pvalues[cluster, gene] = 1.0
        if constant:
            count = float(n_obs)
            tie_term[gene] = count * count * count - count
            continue
        order = np.argsort(column)
        ranks = np.empty(n_obs, dtype=np.float64)
        term = 0.0
        group = 0
        while group < n_obs:
            end = group + 1
            value = column[order[group]]
            while end < n_obs and column[order[end]] == value:
                end += 1
            count = end - group
            average = (group + 1) + (count - 1) / 2.0
            for position in range(group, end):
                ranks[order[position]] = average
            count_f = float(count)
            term += count_f * count_f * count_f - count_f
            group = end
        tie_term[gene] = term
        rank_sum = np.zeros(n_clusters, dtype=np.float64)
        for index in range(n_obs):
            cluster = codes[index]
            if cluster >= 0:
                rank_sum[cluster] += ranks[index]
        n = float(n_obs)
        for cluster in range(n_clusters):
            n1 = n_in[cluster]
            n2 = n - n1
            pvalues[cluster, gene] = _asymptotic_p(rank_sum[cluster], n1, n2, term)


@numba.njit(cache=True)
def _asymptotic_p(rank_sum, n1, n2, tie_term):
    # Same normal approximation as scipy.stats.mannwhitneyu, two-sided, continuity on.
    u1 = rank_sum - n1 * (n1 + 1.0) / 2.0
    u2 = n1 * n2 - u1
    u = u1 if u1 >= u2 else u2
    mu = n1 * n2 / 2.0
    n = n1 + n2
    scale = (n + 1.0) - tie_term / (n * (n - 1.0))
    variance = n1 * n2 / 12.0 * scale
    if variance <= 0.0:
        return 1.0
    z = (u - mu - 0.5) / math.sqrt(variance)
    p = math.erfc(z / math.sqrt(2.0))
    if p > 1.0:
        return 1.0
    if p < 0.0:
        return 0.0
    return p


def _exact_small_clusters(block, codes, n_in, pvalues, tie_term, offset) -> None:
    """Clusters of eight or fewer cells use scipy's exact test when nothing is tied.

    Larger contrasts, and every tied contrast, keep the normal approximation.
    A constant gene stays at p = 1, matching ``measure_gene``.
    """
    n_obs = codes.shape[0]
    n_out = n_obs - n_in
    small = np.flatnonzero((n_in <= 8.0) | (n_out <= 8.0))
    if small.size == 0:
        return
    local = np.flatnonzero(tie_term[offset : offset + block.shape[0]] == 0.0)
    if local.size == 0:
        return
    inside_idx = []
    outside_idx = []
    for cluster in range(n_in.shape[0]):
        inside_idx.append(np.flatnonzero(codes == cluster))
        outside_idx.append(np.flatnonzero(codes != cluster))
    for cluster in small:
        inside_at = inside_idx[cluster]
        outside_at = outside_idx[cluster]
        for local_gene in local:
            column = block[int(local_gene)]
            inside = column[inside_at]
            outside = column[outside_at]
            gene = offset + int(local_gene)
            if (
                inside.size
                and outside.size
                and inside.min() == inside.max()
                and outside.min() == outside.max()
                and inside[0] == outside[0]
            ):
                pvalues[cluster, gene] = 1.0
                continue
            result = stats.mannwhitneyu(inside, outside, alternative="two-sided")
            pvalues[cluster, gene] = float(result.pvalue)


def positive_contribution(gene: GeneEvidence) -> float:
    """A marker counts when its level is higher inside, even if other cells also express it."""
    if gene.logfc <= 0:
        return 0.0
    return gene.logfc


def undeclared_enriched(
    adata: AnnData,
    mask: np.ndarray,
    declared: set[str],
    *,
    limit: int = 10,
    min_logfc: float = 1.0,
) -> list[str]:
    """Genes enriched in the cluster that no hypothesis declared.

    Ranked by log fold change. These names are for inspection only.
    """
    return undeclared_groups(
        adata,
        [np.asarray(mask, dtype=bool)],
        declared,
        limit=limit,
        min_logfc=min_logfc,
    )[0]


def undeclared_groups(
    adata: AnnData,
    masks: list[np.ndarray],
    declared: set[str],
    *,
    limit: int = 10,
    min_logfc: float = 1.0,
) -> list[list[str]]:
    """Top undeclared genes for every mask from one expm1 pass."""
    if not masks:
        return []
    n_obs = adata.n_obs
    codes = np.full(n_obs, -1, dtype=np.int32)
    for cluster, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        count = int(mask.sum())
        if count == 0 or count == n_obs:
            raise ValueError(f"cluster mask {cluster} has no contrast")
        codes[mask] = np.int32(cluster)
    sums, totals = _group_linear_sums(adata.X, codes, len(masks))
    n_in = np.bincount(codes[codes >= 0], minlength=len(masks)).astype(np.float64)
    n_out = n_obs - n_in
    mean_in = sums / n_in[:, None]
    mean_out = (totals - sums) / n_out[:, None]
    changes = np.log2((mean_in + 1.0) / (mean_out + 1.0))
    names = np.asarray(adata.var_names.astype(str))
    blocked = np.fromiter((name in declared for name in names), dtype=bool, count=names.size)
    chosen: list[list[str]] = []
    for cluster in range(len(masks)):
        keep = changes[cluster] >= min_logfc
        keep[blocked] = False
        order = np.argsort(-changes[cluster])
        genes: list[str] = []
        for index in order:
            if not keep[index]:
                continue
            genes.append(str(names[index]))
            if len(genes) >= limit:
                break
        chosen.append(genes)
    return chosen


def _group_linear_sums(matrix, codes: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    n_vars = int(matrix.shape[1])
    if sparse.issparse(matrix):
        stored = matrix.tocsr()
        data = np.expm1(np.asarray(stored.data, dtype=np.float64))
        totals = np.bincount(stored.indices, weights=data, minlength=n_vars).astype(np.float64)
        sums = _sparse_cluster_sums(stored.indptr, stored.indices, data, codes, n_clusters, n_vars)
        return sums, totals
    # Column blocks keep the float64 copy near _MARKER_CHUNK_BYTES.
    n_obs = int(matrix.shape[0])
    step = max(1, _MARKER_CHUNK_BYTES // (8 * max(n_obs, 1)))
    rows = [np.flatnonzero(codes == cluster) for cluster in range(n_clusters)]
    sums = np.zeros((n_clusters, n_vars), dtype=np.float64)
    totals = np.zeros(n_vars, dtype=np.float64)
    for start in range(0, n_vars, step):
        stop = min(n_vars, start + step)
        linear = np.expm1(np.asarray(matrix[:, start:stop], dtype=np.float64))
        totals[start:stop] = linear.sum(axis=0)
        for cluster, index in enumerate(rows):
            sums[cluster, start:stop] = linear[index].sum(axis=0)
    return sums, totals


def _sparse_cluster_sums(indptr, indices, data, codes, n_clusters: int, n_vars: int) -> np.ndarray:
    n_obs = int(codes.shape[0])
    workers = min(_cpus(), max(n_obs, 1))
    partial_bytes = workers * n_clusters * n_vars * 8
    if workers == 1 or n_obs < 2 or partial_bytes > _PARTIAL_SUM_BYTES:
        return _sum_row_range(indptr, indices, data, codes, n_clusters, n_vars, 0, n_obs)
    _sum_row_range(indptr, indices, data, codes, n_clusters, n_vars, 0, 0)
    edges = [int(i * n_obs / workers) for i in range(workers + 1)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(
            pool.map(
                lambda start, stop: _sum_row_range(
                    indptr, indices, data, codes, n_clusters, n_vars, start, stop
                ),
                edges[:-1],
                edges[1:],
            )
        )
    sums = parts[0]
    for part in parts[1:]:
        sums += part
    return sums


@numba.njit(cache=True, nogil=True)
def _sum_row_range(indptr, indices, data, codes, n_clusters, n_vars, start, stop):
    sums = np.zeros((n_clusters, n_vars), dtype=np.float64)
    for row in range(start, stop):
        cluster = codes[row]
        if cluster < 0:
            continue
        for pointer in range(indptr[row], indptr[row + 1]):
            sums[cluster, indices[pointer]] += data[pointer]
    return sums


def negative_marker_conflicts(pct_in: float, pct_out: float, logfc: float) -> bool:
    """True when a negative marker is relatively high inside the cluster.

    Expression in other cell types does not cancel that. ``pct_out`` stays in
    the signature so callers can pass the measured fraction.
    """
    del pct_out
    return pct_in >= NEGATIVE_PRESENT_PCT and logfc >= NEGATIVE_MIN_LOGFC


def negative_penalty(gene: GeneEvidence) -> float:
    """Enriched negative markers penalize. Absence and ambient RNA do not."""
    if not negative_marker_conflicts(gene.pct_in, gene.pct_out, gene.logfc):
        return 0.0
    return gene.logfc
