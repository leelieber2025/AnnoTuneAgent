# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

import json
import sys
from pathlib import Path

import pytest

from annotuneagent.granularity import ontology_dir
from annotuneagent.granularity.cl_dag import OntologyCycleError, load_ontology
from annotuneagent.granularity.ontology_map import load_map
from annotuneagent.granularity.relations import popv_onestep, relation

ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = ROOT / "src" / "annotuneagent" / "ontology"
SHA = "73996c6349283e7a8cbd8183367a1cb810d0077e9c291ae0c72bd31e869f8c1c"


def test_runtime_ontology_uses_packaged_files():
    assert ontology_dir() == ONTOLOGY
    assert (ontology_dir() / "cl_isa.json").is_file()
    assert (ontology_dir() / "ontology_map.jsonl").is_file()


def load_ontology_blob(payload: dict):
    path = Path("/tmp/granularity_blob.json")
    path.write_text(json.dumps(payload))
    return load_ontology(path)


def test_cycle_is_rejected():
    with pytest.raises(OntologyCycleError, match="A -> B -> A"):
        load_ontology_blob(
            {
                "source": {"sha256": "cycle"},
                "labels": {"A": "a", "B": "b"},
                "parents": {"A": ["B"], "B": ["A"]},
            }
        )


def test_shortest_path_uses_the_directed_parent_edges():
    dag = load_ontology_blob(
        {
            "source": {"sha256": "short"},
            "labels": {n: n for n in "ABCDE"},
            "parents": {"A": ["B", "C"], "B": ["D"], "C": ["E"], "E": ["D"]},
        }
    )
    assert dag.up_dist("A", "D") == (2, ["A", "B", "D"])


def _identity_map(dag):
    rows = []
    for node, label in dag.labels.items():
        rows.append(
            {
                "map_version": "t",
                "raw_label": label,
                "species": "*",
                "cl_id": node,
                "cl_label": label,
                "match": "exact",
                "status": "reviewed",
                "reviewer": "test",
                "evidence": "test",
                "ontology": {"name": "t", "sha256": dag.sha256},
            }
        )
    path = Path("/tmp/granularity_identity_map.jsonl")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return load_map(path)


def test_a_shared_ancestor_is_not_enough_for_sibling():
    dag = load_ontology_blob(
        {
            "source": {"sha256": "uncle"},
            "labels": {"C": "C", "P": "P", "U": "U", "G": "G"},
            "parents": {"C": ["P"], "P": ["G"], "U": ["G"]},
        }
    )
    item = relation("C", "U", None, dag, _identity_map(dag))
    assert item["relation"] == "unrelated"
    assert item["popv_onestep"] == "no match"


def test_snapshot_paths_match_the_documented_examples():
    dag = load_ontology(ONTOLOGY / "cl_isa.json")
    label_map = load_map(ONTOLOGY / "ontology_map.jsonl")
    assert dag.sha256 == SHA
    assert "popv" not in sys.modules

    coarse = relation("NK cell", "CD56dim NK cell", "human", dag, label_map)
    assert coarse["relation"] == "ancestor"
    assert coarse["signed_k"] == -2
    assert coarse["path"] == ["CL:0000939", "CL:0000824", "CL:0000623"]
    assert coarse["popv_onestep"] == "no match"

    fine = relation("CD56dim NK cell", "NK cell", "human", dag, label_map)
    assert (fine["relation"], fine["signed_k"]) == ("descendant", 2)
    assert fine["popv_onestep"] == "no match"

    b_cell = relation("B cell", "naive B cell", "human", dag, label_map)
    assert b_cell["signed_k"] == -3
    assert b_cell["path"] == ["CL:0000788", "CL:0000785", "CL:0001201", "CL:0000236"]
    assert b_cell["popv_onestep"] == "no match"

    monocyte = relation("CD14-positive monocyte", "monocyte", "human", dag, label_map)
    assert (monocyte["relation"], monocyte["signed_k"]) == ("descendant", 1)
    assert monocyte["popv_onestep"] == "child"
    assert popv_onestep("CL:0001054", "CL:0000576", dag) == "child"

    naive = relation("naive CD4 T cell", "CD4-positive, alpha-beta T cell", "human", dag, label_map)
    assert (naive["relation"], naive["signed_k"]) == ("descendant", 1)
    assert label_map.lookup("cDC2", "human").cl_id == "CL:0002399"
    assert label_map.lookup("cDC2", "mouse").cl_id == "CL:0002465"
    cd16 = relation("CD16-positive monocyte", "monocyte", "human", dag, label_map)
    assert (cd16["relation"], cd16["signed_k"]) == ("descendant", 2)

    crossed = relation("NK cell", "gamma-delta T cell", "human", dag, label_map)
    assert crossed["relation"] == "unrelated"
    assert crossed["popv_onestep"] == "no match"
    assert dag.lca("CL:0000798", "CL:0000623") == "CL:0000542"

    assert relation("CD56dim NK cell", "NK cell", "mouse", dag, label_map)["relation"] == "unmapped"
    assert relation(None, "NK cell", "human", dag, label_map)["relation"] == "abstain"
    assert label_map.lookup("NK cell", "human", formal=True) is None


def test_same_cl_does_not_count_as_string_exact():
    dag = load_ontology(ONTOLOGY / "cl_isa.json")
    label_map = load_map(ONTOLOGY / "ontology_map.jsonl")
    mapped = relation("NK cell", "NK cell", "human", dag, label_map)
    assert mapped["relation"] == "exact"
    assert mapped["same_cl_diff_raw"] is False
