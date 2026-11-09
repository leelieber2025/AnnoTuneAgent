# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Tool-layer models. Negative-marker penalties stay a hard gate."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .arbitrate import ScoredHypothesis

# Larger means a worse match. A penalized hypothesis is kept out of the set.
_PENALTY_NONCONFORMITY = 1.0e6


@dataclass(frozen=True)
class Ranker:
    """Logistic scores for hypotheses that already passed the negative-marker gate."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coef: tuple[float, ...]
    intercept: float

    def decision(self, values: np.ndarray) -> float:
        scale = np.asarray(self.scale, dtype=float)
        safe = np.where(scale == 0, 1.0, scale)
        standardized = (np.asarray(values, dtype=float) - np.asarray(self.mean)) / safe
        return float(standardized @ np.asarray(self.coef) + self.intercept)


def hypothesis_features(item: ScoredHypothesis) -> np.ndarray:
    """Mean supporting log2 fold-change, mean percent difference, and support fraction."""
    supported = [gene for gene in item.positive if gene.logfc > 0]
    if not supported:
        return np.zeros(3, dtype=float)
    logfc = float(np.mean([gene.logfc for gene in supported]))
    delta = float(np.mean([gene.pct_in - gene.pct_out for gene in supported]))
    fraction = len(supported) / len(item.positive)
    return np.asarray([logfc, delta, fraction], dtype=float)


def fit_ranker(features: list[np.ndarray], labels: list[int]) -> Ranker:
    """Fit a logistic regression. ``labels`` are 1 when the hypothesis is correct."""
    if len(features) != len(labels) or len(features) < 2:
        raise ValueError("need at least two labeled feature rows")
    if len(set(labels)) < 2:
        raise ValueError("ranker labels must include both classes")
    matrix = np.vstack(features)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    standardized = (matrix - mean) / scale
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(standardized, np.asarray(labels, dtype=int))
    return Ranker(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        coef=tuple(float(value) for value in model.coef_.ravel()),
        intercept=float(model.intercept_[0]),
    )


def apply_ranker(scored: list[ScoredHypothesis], ranker: Ranker) -> list[ScoredHypothesis]:
    """Put every score on the ranker scale. A penalized hypothesis ranks below all others.

    Penalized hypotheses keep their order among themselves. Mixing their
    log fold-change scores with ranker scores would make margins meaningless.
    """
    fitted = {
        item.hypothesis.name: ranker.decision(hypothesis_features(item))
        for item in scored
        if item.penalty <= 0
    }
    floor = min(fitted.values(), default=0.0)
    return [
        replace(item, score=fitted[item.hypothesis.name])
        if item.penalty <= 0
        else replace(item, score=floor - item.penalty)
        for item in scored
    ]


def nonconformity(item: ScoredHypothesis, best_other: float) -> float:
    if item.penalty > 0:
        return _PENALTY_NONCONFORMITY
    return float(best_other - item.score)


def calibration_nonconformity(scored: list[ScoredHypothesis], true_name: str) -> float:
    """Nonconformity of the author type on one calibration cluster."""
    match = next((item for item in scored if item.hypothesis.name == true_name), None)
    if match is None:
        raise KeyError(true_name)
    others = [item.score for item in scored if item.hypothesis.name != true_name]
    best_other = max(others) if others else match.score
    return nonconformity(match, best_other)


def conformal_quantile(values: list[float], alpha: float = 0.1) -> float:
    """Upper quantile for a split-conformal prediction set."""
    if not values:
        raise ValueError("calibration nonconformity is empty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    rank = math.ceil((len(ordered) + 1) * (1 - alpha))
    if rank > len(ordered):
        return math.inf
    return ordered[rank - 1]


def prediction_set(scored: list[ScoredHypothesis], quantile: float) -> list[str]:
    """Names whose nonconformity is at most ``quantile``. Penalized names stay out."""
    names: list[str] = []
    for item in scored:
        if item.penalty > 0:
            continue
        others = [other.score for other in scored if other is not item]
        best_other = max(others) if others else item.score
        if nonconformity(item, best_other) <= quantile:
            names.append(item.hypothesis.name)
    return names
