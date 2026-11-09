# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Caller-supplied cell-type hypotheses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hypothesis:
    """One candidate label and the markers that support or oppose it.

    ``positive`` genes should be enriched in the cluster if this label is right.
    Their mean fold change is the score. A gene detected in fewer than 25% of
    the cluster's cells is left out of that mean when at least two genes are
    detected, so an absent gene does not dilute a high one. With fewer than two
    detected, the whole panel is averaged. ``support`` genes add their fold
    change when enriched and do not lower the score when absent. ``negative``
    genes should not be detected. An enriched negative gene lowers the score
    by its fold change, scaled by the number of positive genes averaged.
    Absence of a negative gene is not evidence for the label.
    """

    name: str
    positive: tuple[str, ...]
    negative: tuple[str, ...] = ()
    support: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("hypothesis name is empty")
        if not self.positive:
            raise ValueError(f"{self.name}: at least one positive marker is required")
        object.__setattr__(self, "positive", tuple(self.positive))
        object.__setattr__(self, "negative", tuple(self.negative))
        object.__setattr__(self, "support", tuple(self.support))
