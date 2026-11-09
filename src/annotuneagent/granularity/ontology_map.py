# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Versioned raw-label map. The ontology file itself is not edited."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MATCHES = {"exact", "close", "broader", "narrower", "unmapped"}
STATUSES = {"proposed", "reviewed", "needs_review"}


@dataclass(frozen=True)
class MapEntry:
    map_version: str
    raw_label: str
    species: str
    cl_id: str | None
    cl_label: str | None
    match: str
    status: str
    reviewer: str | None
    evidence: str
    ontology_sha256: str


class OntologyMap:
    def __init__(self, entries: list[MapEntry]):
        if not entries:
            raise ValueError("ontology map is empty")
        versions = {entry.map_version for entry in entries}
        hashes = {entry.ontology_sha256 for entry in entries}
        if len(versions) != 1:
            raise ValueError(f"ontology map has more than one version: {sorted(versions)}")
        if len(hashes) != 1:
            raise ValueError("ontology map rows do not share one snapshot hash")
        self.version = entries[0].map_version
        self.ontology_sha256 = entries[0].ontology_sha256
        self.entries = list(entries)
        self._rows: dict[tuple[str, str, str], MapEntry] = {}
        for entry in entries:
            key = (entry.map_version, entry.raw_label, entry.species)
            if key in self._rows:
                raise ValueError(f"duplicate map key {key}")
            self._rows[key] = entry

    def lookup(
        self, raw_label: str, species: str | None, *, formal: bool = False
    ) -> MapEntry | None:
        for candidate in _species_order(species):
            entry = self._rows.get((self.version, raw_label, candidate))
            if entry is None:
                continue
            if formal and entry.status != "reviewed":
                continue
            return entry
        return None

    @property
    def reviewed_only(self) -> bool:
        return all(entry.status == "reviewed" for entry in self.entries)


def load_map(path: Path) -> OntologyMap:
    entries: list[MapEntry] = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        match = row.get("match")
        status = row.get("status")
        if match not in MATCHES:
            raise ValueError(f"{path}:{lineno}: bad match {match!r}")
        if status not in STATUSES:
            raise ValueError(f"{path}:{lineno}: bad status {status!r}")
        cl_id = row.get("cl_id")
        if match == "unmapped":
            if cl_id:
                raise ValueError(f"{path}:{lineno}: unmapped row has a CL id")
        elif not cl_id:
            raise ValueError(f"{path}:{lineno}: mapped row has no CL id")
        ontology = row.get("ontology") or {}
        entries.append(
            MapEntry(
                map_version=row["map_version"],
                raw_label=row["raw_label"],
                species=row["species"],
                cl_id=cl_id,
                cl_label=row.get("cl_label"),
                match=match,
                status=status,
                reviewer=row.get("reviewer"),
                evidence=row.get("evidence") or "",
                ontology_sha256=ontology.get("sha256") or "",
            )
        )
    return OntologyMap(entries)


def _species_order(species: str | None) -> list[str]:
    order: list[str] = []
    if species and species != "*":
        order.append(species)
    order.append("*")
    return order
