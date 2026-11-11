# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Caller-supplied cell-type hypotheses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hypothesis:
    """One candidate label and the markers that support or oppose it.

    ``positive`` genes should be enriched in the cluster if this label is right.
    Their mean fold change is the score. Every positive gene stays in that
    mean. A gene detected in fewer than 25% of the cluster's cells contributes
    0, so a missing identity gene dilutes genes that are shared with other
    types. ``support`` genes add their fold change when enriched and do not
    lower the score when absent. ``negative`` genes should not be detected. An
    enriched negative gene is a reference: it lowers the score by half of its
    fold change divided by the number of positive genes. Absence of a negative
    gene is not evidence for the label. ``requires_any`` lists genes of which at
    least one must be detected in 25% of the cluster. Otherwise this name cannot win.
    """

    name: str
    positive: tuple[str, ...]
    negative: tuple[str, ...] = ()
    support: tuple[str, ...] = ()
    requires_any: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # A bare string is one gene. tuple("CD3E") would split it into letters.
        for field in ("positive", "negative", "support", "requires_any"):
            value = getattr(self, field)
            if isinstance(value, str):
                object.__setattr__(self, field, (value,) if value else ())
        if not self.name:
            raise ValueError("hypothesis name is empty")
        if not self.positive:
            raise ValueError(f"{self.name}: at least one positive marker is required")
        object.__setattr__(self, "positive", tuple(self.positive))
        object.__setattr__(self, "negative", tuple(self.negative))
        object.__setattr__(self, "support", tuple(self.support))
        object.__setattr__(self, "requires_any", tuple(self.requires_any))
