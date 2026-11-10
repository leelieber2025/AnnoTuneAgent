# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Turn gene evidence into one cluster record. This is not a model tool."""

from __future__ import annotations

from dataclasses import dataclass

from .evidence import NEGATIVE_PRESENT_PCT, GeneEvidence, negative_penalty, positive_contribution
from .hypotheses import Hypothesis

# Score gap below this refuses a hard label.
DEFAULT_MIN_MARGIN = 0.15
# A positive panel is on when at least half of its genes are detected in the cluster.
_PROGRAM_PCT = 0.5


@dataclass(frozen=True)
class ScoredHypothesis:
    hypothesis: Hypothesis
    positive: tuple[GeneEvidence, ...]
    negative: tuple[GeneEvidence, ...]
    score: float
    penalty: float
    eligible: bool = True

    def as_evidence(self) -> dict[str, object]:
        return {
            "positive": [gene.as_dict() for gene in self.positive],
            "negative": [gene.as_dict() for gene in self.negative],
        }


def score_hypothesis(
    hypothesis: Hypothesis,
    positive: list[GeneEvidence],
    negative: list[GeneEvidence],
    support: list[GeneEvidence] | None = None,
    required: list[GeneEvidence] | None = None,
) -> ScoredHypothesis:
    # Every positive gene stays in the mean. A gene detected in fewer than 25%
    # of the cells contributes 0, so a missing identity gene dilutes genes that
    # are shared with other types. One detected gene cannot stand for the panel.
    parts = [
        positive_contribution(gene) if gene.pct_in >= NEGATIVE_PRESENT_PCT else 0.0
        for gene in positive
    ]
    score = float(sum(parts) / len(parts)) if parts else 0.0
    score += float(sum(positive_contribution(gene) for gene in support or []))
    fired = [value for value in (negative_penalty(gene) for gene in negative) if value > 0]
    # One opposing gene weighs like one positive gene. Subtracting the whole
    # logfc let a single shared gene erase a specific panel.
    penalty = float(sum(fired) / len(parts)) if fired and parts else float(sum(fired))
    # None means this name has no required gene. An empty list means the
    # required genes were absent from the matrix, so the name cannot win.
    eligible = (
        True if required is None else any(gene.pct_in >= NEGATIVE_PRESENT_PCT for gene in required)
    )
    return ScoredHypothesis(
        hypothesis=hypothesis,
        positive=tuple(positive),
        negative=tuple(negative),
        score=score - penalty,
        penalty=penalty,
        eligible=eligible,
    )


def program_detected(genes: tuple[GeneEvidence, ...] | list[GeneEvidence]) -> bool:
    """True when two positive genes, and at least half the panel, are detected.

    One detected gene is not a second program. A two-gene panel therefore needs
    both genes, including when the other gene is ambient.
    """
    if len(genes) < 2:
        return False
    detected = sum(gene.pct_in >= _PROGRAM_PCT for gene in genes)
    return detected >= 2 and detected * 2 >= len(genes)


def arbitrate(
    cluster: str,
    scored: list[ScoredHypothesis],
    *,
    min_margin: float = DEFAULT_MIN_MARGIN,
    de_not_in_markers: list[str] | None = None,
    conformal_q: float | None = None,
) -> dict[str, object]:
    if len(scored) < 1:
        raise ValueError("no hypotheses to arbitrate")
    ranked = sorted(scored, key=lambda item: (-item.score, item.hypothesis.name))
    eligible = [item for item in ranked if item.eligible]
    # A name that failed its required genes stays visible, after the names that can win.
    ordered = [*eligible, *[item for item in ranked if not item.eligible]]
    top = eligible[0] if eligible else None
    runner = eligible[1] if len(eligible) > 1 else None
    margin = 0.0 if top is None else top.score if runner is None else top.score - runner.score
    accepted: list[str] | None = None
    if conformal_q is not None and eligible:
        from .mltools import prediction_set

        accepted = prediction_set(eligible, conformal_q)
    status = "label"
    blocked = top is None or top.penalty > 0
    blocked = blocked or not any(positive_contribution(gene) > 0 for gene in top.positive)
    if blocked:
        status = "uncertain"
    elif accepted is not None:
        status = "label" if accepted == [top.hypothesis.name] else "uncertain"
    elif runner is not None and margin < min_margin:
        status = "uncertain"
    cell_type = None if top is None or status != "label" else top.hypothesis.name
    reasoning = _reasoning(top if top is not None else ranked[0], status)
    return {
        "cluster": str(cluster),
        "status": status,
        "cell_type": cell_type,
        "margin": margin,
        "score": 0.0 if top is None else top.score,
        "runner_up": None if runner is None else runner.hypothesis.name,
        "reasoning": reasoning,
        "evidence": {
            **(ranked[0] if top is None else top).as_evidence(),
            "de_not_in_markers": list(de_not_in_markers or []),
        },
        "ranking": [
            {
                "name": item.hypothesis.name,
                "score": item.score,
                "program_on": program_detected(item.positive),
                "eligible": item.eligible,
            }
            for item in ordered
        ],
        "prediction_set": accepted,
        "hypotheses_tested": [item.hypothesis.name for item in ranked],
        "used_tools": ["score_positive_markers", "score_negative_markers", "run_de"],
    }


def _reasoning(top: ScoredHypothesis, status: str) -> str:
    up = [gene.gene for gene in top.positive if positive_contribution(gene) > 0]
    against = [gene.gene for gene in top.negative if negative_penalty(gene) > 0]
    parts: list[str] = []
    if up:
        parts.append(f"{', '.join(up)} enriched in the cluster")
    if against:
        parts.append(f"{', '.join(against)} enriched despite being negative markers")
    if not parts:
        parts.append("no positive enrichment")
    sentence = "; ".join(parts) + "."
    if status == "uncertain":
        return sentence + " No hard label."
    return sentence
