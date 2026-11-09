# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Cell Ontology lineage used when a written name is chosen.

The snapshot is packaged data. popV is not imported and is not a dependency.
"""

from pathlib import Path

from .cl_dag import CellOntology, OntologyCycleError, load_ontology
from .ontology_map import OntologyMap, load_map
from .relations import relation


def ontology_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "ontology"


__all__ = [
    "CellOntology",
    "OntologyCycleError",
    "OntologyMap",
    "load_map",
    "load_ontology",
    "ontology_dir",
    "relation",
]
