# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Built-in marker dictionary.

Rows live in ``markers.tsv``. A row is one species and either ``any`` tissue
or one tissue. Gene symbols are the ones used in that species' matrix.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .hypotheses import Hypothesis

SPECIES = ("human", "mouse")
_TABLE = Path(__file__).with_name("markers.tsv")
# Author phrases that are not themselves a page name. The caller still chooses.
_AUTHOR_TISSUE = (
    ("dorsal root ganglion", "drg"),
    ("dorsal root ganglia", "drg"),
    ("corneal", "cornea"),
    ("retinal", "retina"),
    ("placental", "placenta"),
    ("thymic", "thymus"),
    ("endometrial", "endometrium"),
    ("ovarian", "ovary"),
    ("testicular", "testis"),
    ("breast", "mammary"),
)


@dataclass(frozen=True)
class MarkerEntry:
    species: str
    scope: str
    cell_type: str
    parent: str | None
    positive: tuple[str, ...]
    negative: tuple[str, ...]
    support: tuple[str, ...]
    note: str
    evidence: str


def _split_genes(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(";") if part.strip())


def load_entries() -> list[MarkerEntry]:
    entries: list[MarkerEntry] = []
    with _TABLE.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            species = row["species"].strip()
            scope = row["scope"].strip()
            if species not in SPECIES:
                raise ValueError(f"unknown species {species!r} in {_TABLE.name}")
            if not scope:
                raise ValueError("empty scope")
            parent = row["parent"].strip() or None
            entries.append(
                MarkerEntry(
                    species=species,
                    scope=scope,
                    cell_type=row["cell_type"].strip(),
                    parent=parent,
                    positive=_split_genes(row["positive"]),
                    negative=_split_genes(row["negative"]),
                    support=_split_genes(row.get("support") or ""),
                    note=row.get("note", "").strip(),
                    evidence=row.get("evidence", "").strip(),
                )
            )
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (entry.species, entry.cell_type)
        if key in seen:
            raise ValueError(f"duplicate marker row for {key}")
        if not entry.positive:
            raise ValueError(f"{entry.cell_type}: no positive markers")
        seen.add(key)
    # A parent is another row of the same species. A removed umbrella type must not linger here.
    for entry in entries:
        if entry.parent is not None and (entry.species, entry.parent) not in seen:
            raise ValueError(
                f"{entry.cell_type}: parent {entry.parent!r} is not a {entry.species} row"
            )
    return entries


_ENTRIES = load_entries()


def _available(species: str, tissue: str | None) -> list[MarkerEntry]:
    _check_species(species)
    if tissue is not None and tissue not in tissue_pages(species):
        raise ValueError(
            f"unknown {species} tissue {tissue!r}; choose from tissue_pages({species!r})"
        )
    return [
        entry
        for entry in _ENTRIES
        if entry.species == species and (entry.scope == "any" or entry.scope == tissue)
    ]


def _check_species(species: str) -> None:
    if species not in SPECIES:
        raise ValueError(f"species must be one of {', '.join(SPECIES)}")


def tissue_pages(species: str | None = None) -> list[str]:
    """Tissue pages that can be added to the core dictionary.

    The caller chooses one page and passes it as ``tissue``. Pages are listed
    in table order.
    """
    if species is not None:
        _check_species(species)
    pages: list[str] = []
    for entry in _ENTRIES:
        if species is not None and entry.species != species:
            continue
        if entry.scope == "any" or entry.scope in pages:
            continue
        pages.append(entry.scope)
    return pages


def suggest_tissues(author: str, species: str | None = None) -> list[str]:
    """Pages consistent with an author tissue phrase.

    The caller chooses one and passes it to ``run``. An unmatched phrase
    returns every page. Dorsal root ganglion returns the ``drg`` page.
    """
    pages = tissue_pages(species)
    text = " ".join(author.casefold().replace("_", " ").replace("-", " ").split())
    tokens = set(re.findall(r"[a-z0-9]+", text))
    hits: list[str] = []
    for phrase, page in _AUTHOR_TISSUE:
        if phrase in text and page in pages and page not in hits:
            hits.append(page)
    for page in pages:
        if page in tokens and page not in hits:
            hits.append(page)
    if not hits:
        return pages
    return [page for page in pages if page in set(hits)]


def marker_names(species: str, tissue: str | None = None) -> list[str]:
    """Types that can be scored for this species and optional tissue."""
    return [entry.cell_type for entry in _available(species, tissue)]


def hypotheses_from_names(
    names: list[str],
    species: str,
    tissue: str | None = None,
) -> list[Hypothesis]:
    """Look up hypotheses. Tissue rows are visible only when ``tissue`` matches."""
    by_name = {entry.cell_type: entry for entry in _available(species, tissue)}
    missing = [name for name in names if name not in by_name]
    if missing:
        where = _explain_missing(missing[0], species, tissue)
        raise KeyError(where)
    return [
        Hypothesis(
            name,
            positive=by_name[name].positive,
            negative=by_name[name].negative,
            support=by_name[name].support,
        )
        for name in names
    ]


def _explain_missing(name: str, species: str, tissue: str | None) -> str:
    other = [
        entry
        for entry in _ENTRIES
        if entry.cell_type == name and entry.species == species and entry.scope != "any"
    ]
    if other and tissue != other[0].scope:
        return f"{name!r} is a {species} {other[0].scope} type; pass tissue={other[0].scope!r}"
    known = ", ".join(marker_names(species, tissue))
    return f"unknown cell type {name!r} for {species}. Known names: {known}"


def parents_for(
    names: list[str] | None = None,
    *,
    species: str,
    tissue: str | None = None,
) -> dict[str, str]:
    """Map a type to its parent when the selected rows define one."""
    chosen = marker_names(species, tissue) if names is None else names
    by_name = {entry.cell_type: entry for entry in _available(species, tissue)}
    return {
        name: by_name[name].parent
        for name in chosen
        if name in by_name and by_name[name].parent is not None
    }
