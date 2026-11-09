# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Directed Cell Ontology relation between two labels."""

from __future__ import annotations

from .cl_dag import CellOntology
from .ontology_map import OntologyMap


def popv_onestep(pred: str, gold: str, dag: CellOntology) -> str:
    """popV `_ontology_accuracy` order: exact, sibling, parent, child, no match.

    `parent` means the prediction is the gold label's direct parent (one step coarser).
    `child` means the gold label is the prediction's direct parent (one step finer).
    """
    if pred == gold:
        return "exact"
    pred_parents = dag.parents_of(pred)
    gold_parents = dag.parents_of(gold)
    if pred_parents & gold_parents:
        return "sibling"
    if pred in gold_parents:
        return "parent"
    if gold in pred_parents:
        return "child"
    return "no match"


def relation(
    pred_raw: str | None,
    gold_raw: str | None,
    species: str | None,
    dag: CellOntology,
    label_map: OntologyMap,
    *,
    formal: bool = False,
) -> dict:
    """Relation of the prediction `pred_raw` to the reference `gold_raw`.

    Ancestor means the prediction is coarser (negative k). Descendant means finer.
    Distance is directed along is_a. Unmapped labels are not guessed.
    """
    if pred_raw is None or gold_raw is None or pred_raw == "" or gold_raw == "":
        return _blank("abstain")
    pred_entry = label_map.lookup(pred_raw, species, formal=formal)
    gold_entry = label_map.lookup(gold_raw, species, formal=formal)
    pred_id = pred_entry.cl_id if pred_entry is not None else None
    gold_id = gold_entry.cl_id if gold_entry is not None else None
    if pred_id is None or gold_id is None:
        return _blank("unmapped")
    step = popv_onestep(pred_id, gold_id, dag)
    if pred_id == gold_id:
        return {
            "relation": "exact",
            "signed_k": 0,
            "path": [pred_id],
            "popv_onestep": step,
            "same_cl_diff_raw": pred_raw != gold_raw,
        }
    if pred_id in dag.ancestors(gold_id):
        found = dag.up_dist(gold_id, pred_id)
        if found is None:
            raise RuntimeError(f"missing upward path {gold_id} -> {pred_id}")
        dist, path = found
        return _row("ancestor", -dist, path, step)
    if gold_id in dag.ancestors(pred_id):
        found = dag.up_dist(pred_id, gold_id)
        if found is None:
            raise RuntimeError(f"missing upward path {pred_id} -> {gold_id}")
        dist, path = found
        return _row("descendant", dist, path, step)
    if dag.parents_of(pred_id) & dag.parents_of(gold_id):
        return _row("sibling", None, None, step)
    return _row("unrelated", None, None, step)


def _blank(kind: str) -> dict:
    return {
        "relation": kind,
        "signed_k": None,
        "path": None,
        "popv_onestep": None,
        "same_cl_diff_raw": False,
    }


def _row(kind: str, signed_k: int | None, path: list[str] | None, step: str) -> dict:
    return {
        "relation": kind,
        "signed_k": signed_k,
        "path": path,
        "popv_onestep": step,
        "same_cl_diff_raw": False,
    }
