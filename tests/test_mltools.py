import numpy as np

from annotuneagent.arbitrate import ScoredHypothesis, arbitrate
from annotuneagent.evidence import GeneEvidence
from annotuneagent.hypotheses import Hypothesis
from annotuneagent.mltools import (
    apply_ranker,
    calibration_nonconformity,
    conformal_quantile,
    fit_ranker,
    hypothesis_features,
    prediction_set,
)


def _gene(name: str, logfc: float, pct_in: float, pct_out: float, direction: str) -> GeneEvidence:
    return GeneEvidence(
        gene=name,
        direction=direction,
        pct_in=pct_in,
        pct_out=pct_out,
        mean_in=logfc,
        mean_out=0.0,
        logfc=logfc,
        pvalue=0.01,
        expressed=pct_in >= 0.25,
    )


def _scored(name: str, logfc: float, penalty: float) -> ScoredHypothesis:
    positive = (_gene("G", logfc, 0.8, 0.1, "positive"),)
    return ScoredHypothesis(
        hypothesis=Hypothesis(name, positive=("G",), negative=()),
        positive=positive,
        negative=(),
        score=logfc - penalty,
        penalty=penalty,
    )


def test_ranker_ranks_penalized_hypotheses_below_all_others():
    rows = [np.array([2.0, 0.7, 1.0]), np.array([0.1, 0.0, 0.2])] * 4
    labels = [1, 0] * 4
    ranker = fit_ranker(rows, labels)
    penalized = _scored("bad", 0.2, penalty=1.0)
    clean = _scored("good", 2.0, penalty=0.0)
    updated = apply_ranker([penalized, clean], ranker)
    assert updated[0].penalty == 1.0
    assert updated[1].score == ranker.decision(hypothesis_features(clean))
    assert updated[0].score == updated[1].score - penalized.penalty
    assert updated[1].score > updated[0].score


def test_ranker_penalized_alone_keeps_a_finite_score():
    rows = [np.array([2.0, 0.7, 1.0]), np.array([0.1, 0.0, 0.2])] * 4
    ranker = fit_ranker(rows, [1, 0] * 4)
    only = apply_ranker([_scored("bad", 0.2, penalty=1.0)], ranker)
    assert only[0].score == -1.0


def test_penalized_hypothesis_stays_out_when_the_quantile_is_infinite():
    clean = _scored("a", 1.0, 0.0)
    blocked = _scored("b", 3.0, 3.0)
    quantile = conformal_quantile([0.1, 0.2], alpha=0.1)
    assert quantile == float("inf")
    assert prediction_set([clean, blocked], quantile) == ["a"]


def test_conformal_set_shrinks_when_the_quantile_is_tight():
    wide = _scored("a", 1.0, 0.0)
    close = _scored("b", 0.9, 0.0)
    blocked = _scored("c", 3.0, 1.0)
    assert "c" not in prediction_set([wide, close, blocked], quantile=10)
    assert prediction_set([wide, close], quantile=-0.05) == ["a"]
    assert set(prediction_set([wide, close], quantile=0.2)) == {"a", "b"}
    assert conformal_quantile([-0.2, -0.1, 0.0], alpha=0.34) == 0.0


def test_calibration_uses_the_true_type_and_arbitrate_requires_a_singleton():
    top = _scored("a", 1.0, 0.0)
    other = _scored("b", 0.2, 0.0)
    assert calibration_nonconformity([top, other], "a") == other.score - top.score
    record = arbitrate("0", [top, other], conformal_q=-0.5)
    assert record["status"] == "label"
    assert record["prediction_set"] == ["a"]
    wide = arbitrate("0", [top, other], conformal_q=0.9)
    assert wide["status"] == "uncertain"
    assert wide["cell_type"] is None
