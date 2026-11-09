# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Fixed cluster annotation loop."""

from __future__ import annotations

from collections.abc import Collection

from anndata import AnnData

from .arbitrate import DEFAULT_MIN_MARGIN, arbitrate, score_hypothesis
from .evidence import (
    cluster_ids,
    cluster_mask,
    evidence_from_stats,
    marker_statistics,
    undeclared_groups,
    validate_expression,
)
from .hypotheses import Hypothesis


def annotate(
    adata: AnnData,
    cluster_key: str,
    hypotheses: list[Hypothesis],
    *,
    min_margin: float = DEFAULT_MIN_MARGIN,
    ranker: object | None = None,
    conformal_q: float | None = None,
    clusters: Collection[str] | None = None,
) -> list[dict[str, object]]:
    """Score each cluster against ``hypotheses`` and return one record per cluster.

    ``clusters`` limits the records to those cluster ids. Each cluster is still
    compared with every other cell, so its record is the same either way.

    Expression is read from ``adata.X``. Genes named by a hypothesis must be in
    ``var_names``. Declared markers are scored together across clusters, using
    every CPU. Mann-Whitney p-values are reported and do not change the score.
    """
    if adata.n_obs == 0:
        raise ValueError("AnnData has no cells")
    if not hypotheses:
        raise ValueError("at least one hypothesis is required")
    validate_expression(adata)
    ids = cluster_ids(adata, cluster_key)
    if clusters is not None:
        wanted = {str(item) for item in clusters}
        ids = [item for item in ids if item in wanted]
    masks = [cluster_mask(adata, cluster_key, cluster) for cluster in ids]
    genes = list(
        dict.fromkeys(
            gene
            for hypothesis in hypotheses
            for gene in (*hypothesis.positive, *hypothesis.negative, *hypothesis.support)
        )
    )
    measured = marker_statistics(adata, masks, genes)
    declared = set(genes)
    extras = undeclared_groups(adata, masks, declared)
    records: list[dict[str, object]] = []
    for cluster, cluster_stats, extra in zip(ids, measured, extras, strict=True):
        scored = []
        for hypothesis in hypotheses:
            positive = [
                evidence_from_stats(gene, "positive", cluster_stats[gene])
                for gene in hypothesis.positive
            ]
            negative = [
                evidence_from_stats(gene, "negative", cluster_stats[gene])
                for gene in hypothesis.negative
            ]
            support = [
                evidence_from_stats(gene, "positive", cluster_stats[gene])
                for gene in hypothesis.support
            ]
            scored.append(score_hypothesis(hypothesis, positive, negative, support))
        if ranker is not None:
            from .mltools import apply_ranker

            scored = apply_ranker(scored, ranker)
        records.append(
            arbitrate(
                cluster,
                scored,
                min_margin=min_margin,
                de_not_in_markers=extra,
                conformal_q=conformal_q,
            )
        )
    return records
