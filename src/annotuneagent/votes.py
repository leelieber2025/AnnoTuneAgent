# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Agree a cluster label from several views of the same evidence.

A view is one name or nothing. Three names that share a parent are enough to
write that parent. Three that name the same child write the child. One
dissenting view does not block. Two coarse names with no three-vote parent
stay unresolved.
"""

from __future__ import annotations

from collections import Counter

from .evidence import NEGATIVE_MIN_LOGFC, NEGATIVE_PRESENT_PCT
from .hypotheses import Hypothesis

VIEWS = ("panel", "genes", "subcluster", "neighbors")


def _open_rows(ranking: list[dict]) -> list[dict]:
    """Rows the receptor gate still allows. A missing flag stays open."""
    return [row for row in ranking if row.get("eligible") is not False]


def panel_vote(record: dict) -> str | None:
    """Top dictionary type when its positives are high and negatives are not."""
    ranking = list(record.get("ranking") or [])
    if record.get("status") == "label" and record.get("cell_type"):
        name = str(record["cell_type"])
        blocked = any(
            str(row.get("name")) == name and row.get("eligible") is False for row in ranking
        )
        if not blocked:
            return name
    ranking = _open_rows(ranking)
    if not ranking or float(ranking[0]["score"]) <= 0:
        return None
    for gene in (record.get("evidence") or {}).get("negative") or []:
        if (
            gene.get("pct_in", 0) >= NEGATIVE_PRESENT_PCT
            and gene.get("logfc", 0) >= NEGATIVE_MIN_LOGFC
        ):
            return None
    return str(ranking[0]["name"])


def gene_vote(hypotheses: list[Hypothesis], logfc: dict[str, float]) -> str | None:
    """Dictionary type whose measured positive markers are relatively high."""
    scored: list[tuple[int, float, str]] = []
    for item in hypotheses:
        if item.requires_any and not any(logfc.get(gene, 0.0) > 0 for gene in item.requires_any):
            continue
        present = [gene for gene in item.positive if gene in logfc]
        high = [gene for gene in present if logfc[gene] > 0]
        if not high:
            continue
        if len(present) >= 2 and len(high) < 2:
            continue
        scored.append((len(high), sum(logfc[gene] for gene in high), item.name))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    if len(scored) > 1 and scored[0][:2] == scored[1][:2]:
        return None
    return scored[0][2]


def subcluster_vote(children: list[dict]) -> str | None:
    """Name shared by every labeled child. A split casts no single vote."""
    if not children or any(row.get("status") != "label" for row in children):
        return None
    names = {str(row["cell_type"]) for row in children}
    if len(names) == 1:
        return next(iter(names))
    return None


def agreed_name(votes: dict[str, str | None], parents: dict[str, str]) -> str | None:
    """Parent first, then a child when three votes name that child."""
    named = [name for name in votes.values() if name]
    if len(named) < 3:
        return None

    def coarse(name: str) -> str:
        return parents.get(name, name)

    parent, parent_votes = Counter(coarse(name) for name in named).most_common(1)[0]
    if parent_votes < 3:
        return None
    fine_counts = Counter(name for name in named if coarse(name) == parent)
    fine, fine_votes = fine_counts.most_common(1)[0]
    if fine_votes >= 3:
        return fine
    return parent
