# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Cluster-level comparison against author labels.

Counts are the result. A rate that drops ``uncertain`` clusters is not reported,
because refusing a label is part of the outcome.
"""

from __future__ import annotations

from collections import Counter

from anndata import AnnData

from .evidence import cluster_ids


def majority_label(labels: list[str]) -> str | None:
    """Most common label. Equal top counts mean the reference is ambiguous."""
    if not labels:
        return None
    counts = Counter(labels)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def _coarse(predicted: str, reference: str, parents: dict[str, str]) -> bool:
    if predicted == reference:
        return True
    if parents.get(predicted) == reference:
        return True
    if parents.get(reference) == predicted:
        return True
    return False


def evaluate(
    records: list[dict[str, object]],
    adata: AnnData,
    cluster_key: str,
    label_key: str,
    *,
    parents: dict[str, str] | None = None,
    prediction_key: str | None = None,
) -> dict[str, object]:
    """Compare one record per cluster with the majority author label in that cluster.

    ``parents`` maps a label to the coarser label that still counts as a match.
    Sibling types that share a parent do not match each other. When a workflow
    wrote cell labels, ``prediction_key`` evaluates those final labels.
    """
    if label_key not in adata.obs.columns:
        raise KeyError(f"obs has no column {label_key!r}")
    if prediction_key is not None and prediction_key not in adata.obs.columns:
        raise KeyError(f"obs has no column {prediction_key!r}")
    parent_map = {} if parents is None else parents
    by_cluster = {str(record["cluster"]): record for record in records}
    n_exact = 0
    n_coarse = 0
    n_ambiguous = 0
    n_label = 0
    n_uncertain = 0
    n_mixed = 0
    per_cluster: list[dict[str, object]] = []
    # A cell with no author label takes no part in the majority.
    author = adata.obs[label_key]
    known = author.notna()
    groups = adata.obs[cluster_key].astype(str)
    for cluster in cluster_ids(adata, cluster_key):
        record = by_cluster[cluster]
        status = str(record["status"])
        predicted = record["cell_type"]
        if prediction_key is not None:
            labels = adata.obs.loc[groups == cluster, prediction_key]
            names = labels.dropna().astype(str).unique()
            if len(names) == 1 and not labels.isna().any():
                status, predicted = "label", str(names[0])
            elif len(names) == 0:
                status, predicted = "uncertain", None
            else:
                status, predicted = "mixed", None
        if status == "label":
            n_label += 1
        elif status == "uncertain":
            n_uncertain += 1
        elif status == "mixed":
            n_mixed += 1
        reference = majority_label(author[(groups == cluster) & known].astype(str).tolist())
        exact = False
        coarse = False
        if reference is None:
            n_ambiguous += 1
        elif status == "label" and isinstance(predicted, str):
            exact = predicted == reference
            coarse = _coarse(predicted, reference, parent_map)
            n_exact += int(exact)
            n_coarse += int(coarse)
        per_cluster.append(
            {
                "cluster": cluster,
                "status": status,
                "predicted": predicted,
                "reference": reference,
                "exact": exact,
                "coarse": coarse,
            }
        )
    n_clusters = len(per_cluster)
    return {
        "n_clusters": n_clusters,
        "n_label": n_label,
        "n_uncertain": n_uncertain,
        "n_mixed": n_mixed,
        "n_ambiguous_reference": n_ambiguous,
        "n_exact": n_exact,
        "n_coarse": n_coarse,
        "clusters": per_cluster,
    }
