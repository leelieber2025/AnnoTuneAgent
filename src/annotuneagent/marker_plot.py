# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Literature backed marker panels checked against the current expression matrix."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy import sparse

from .evidence import measure_gene
from .markers import load_entries

MIN_PCT_IN = 0.25
MIN_LOGFC = 0.5
MAX_PVALUE = 0.05
MAX_MARKERS_PER_TYPE = 4


def select_marker_panels(
    adata: AnnData, label_key: str, species: str | None, tissue: str | None
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, object]]], dict[str, str]]:
    """Keep cited positive and support markers that are expressed and enriched.

    A type with no passing marker is still drawn with its best-measured cited
    genes; its status stays ``no_marker_passed_expression_check``.

    The specificity contrast uses the strongest *other labeled type* so panels
    form diagonal blocks in a dotplot. The p-value tests the type against all
    other cells, including cells without a written label.
    """
    labels = adata.obs[label_key]
    names = list(dict.fromkeys(labels.dropna().astype(str)))
    panels: dict[str, list[str]] = {}
    checks: dict[str, list[dict[str, object]]] = {}
    status: dict[str, str] = {}
    entries = {
        entry.cell_type: entry
        for entry in load_entries()
        if species == entry.species
        and (entry.scope == "any" or entry.scope == tissue)
        and "lit=PMID:" in entry.evidence
    }
    observed = set(map(str, adata.var_names))
    # StringDtype NA stays NA through astype(str); a later equality is not a mask.
    label_text = labels.astype("string").fillna("").to_numpy()
    position = {name: i for i, name in enumerate(names)}
    type_codes = np.array([position.get(x, -1) for x in label_text])
    type_sizes = np.bincount(type_codes[type_codes >= 0], minlength=len(names))
    for name in names:
        entry = entries.get(name)
        if entry is None:
            checks[name] = []
            status[name] = "no_literature_backed_dictionary_row"
            continue
        mask = label_text == name
        if mask.all():
            checks[name] = []
            status[name] = "no_other_cells_for_expression_test"
            continue
        rows: list[dict[str, object]] = []
        pmids = re.findall(r"PMID:\d+", entry.evidence)
        cited = (*entry.positive, *entry.support)
        n_tests = sum(gene in observed for gene in cited)
        for gene in cited:
            if gene not in observed:
                rows.append(
                    {
                        "gene": gene,
                        "literature": entry.evidence,
                        "pmids": pmids,
                        "selected": False,
                        "reason": "not_measured",
                    }
                )
                continue
            evidence = measure_gene(adata, mask, gene, direction="positive")
            column = adata[:, gene].X
            values = (
                column.toarray().ravel() if sparse.issparse(column) else np.asarray(column).ravel()
            )
            linear = np.expm1(values)
            kept = type_codes >= 0
            sums = np.bincount(type_codes[kept], weights=linear[kept], minlength=len(names))
            means = sums / np.maximum(type_sizes, 1)
            competitor_means = [float(m) for i, m in enumerate(means) if names[i] != name]
            strongest_other = max(competitor_means, default=evidence.mean_out)
            specificity_logfc = float(np.log2((evidence.mean_in + 1) / (strongest_other + 1)))
            adjusted_pvalue = (
                min(1.0, evidence.pvalue * n_tests) if evidence.pvalue is not None else None
            )
            reasons = []
            if evidence.pct_in < MIN_PCT_IN:
                reasons.append("low_detection")
            if evidence.logfc < MIN_LOGFC:
                reasons.append("not_enriched_vs_all_cells")
            if specificity_logfc < MIN_LOGFC:
                reasons.append("not_specific_vs_other_labels")
            if adjusted_pvalue is None or adjusted_pvalue > MAX_PVALUE:
                reasons.append("not_significant")
            rows.append(
                {
                    "gene": gene,
                    "literature": entry.evidence,
                    "pmids": pmids,
                    "pct_in": evidence.pct_in,
                    "mean_in": evidence.mean_in,
                    "mean_out": evidence.mean_out,
                    "logfc": evidence.logfc,
                    "strongest_other_mean": strongest_other,
                    "specificity_logfc": specificity_logfc,
                    "pvalue": evidence.pvalue,
                    "adjusted_pvalue": adjusted_pvalue,
                    "selected": False,
                    "reason": "+".join(reasons) if reasons else None,
                }
            )
        eligible = [row for row in rows if row["reason"] is None]
        eligible.sort(key=lambda row: (-row["specificity_logfc"], -row["pct_in"], row["gene"]))
        chosen = eligible[:MAX_MARKERS_PER_TYPE]
        for row in chosen:
            row["selected"] = True
        if chosen:
            panels[name] = [str(row["gene"]) for row in chosen]
            status[name] = "selected"
        else:
            status[name] = "no_marker_passed_expression_check"
            # The type is still drawn, with its best cited genes, so the figure
            # shows what was checked. These rows stay selected=False.
            measured = [row for row in rows if "specificity_logfc" in row]
            measured.sort(key=lambda row: (-row["specificity_logfc"], -row["pct_in"], row["gene"]))
            if measured:
                panels[name] = [str(row["gene"]) for row in measured[:MAX_MARKERS_PER_TYPE]]
        checks[name] = rows
    return panels, checks, status


def plot_marker_panels(
    adata: AnnData,
    label_key: str,
    panels: dict[str, list[str]],
    order: list[str],
    path: Path | str | None = None,
) -> str | None:
    """Plot cited, data checked markers with matching type and marker order."""
    if not panels:
        return None
    part = adata[adata.obs[label_key].notna()].copy()
    ordered_panels = {name: panels[name] for name in order if name in panels}
    plot = sc.pl.dotplot(
        part,
        ordered_panels,
        groupby=label_key,
        categories_order=order,
        dendrogram=False,
        use_raw=False,
        standard_scale="var",
        show=False,
        return_fig=True,
    )
    if path is None:
        plot.show()
        return None
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plot.savefig(str(target))
    return str(target)
