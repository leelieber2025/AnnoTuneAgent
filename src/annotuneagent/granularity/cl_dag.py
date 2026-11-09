# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Cell Ontology DAG: is_a edges, cycle check, shortest upward distance, LCA."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path


class OntologyCycleError(RuntimeError):
    def __init__(self, cycle: list[str]):
        self.cycle = cycle
        shown = " -> ".join(cycle)
        super().__init__(f"ontology has a cycle: {shown}")


class CellOntology:
    def __init__(self, parents: dict[str, set[str]], labels: dict[str, str], source: dict):
        self.labels = dict(labels)
        self.parents = {node: set(items) for node, items in parents.items()}
        self.source = dict(source)
        nodes = set(self.parents)
        for items in self.parents.values():
            nodes.update(items)
        self.nodes = nodes
        self._ancestors: dict[str, set[str]] = {}
        self._depth: dict[str, int | None] = {}
        _reject_cycles(self.parents, nodes)

    @property
    def sha256(self) -> str:
        return str(self.source.get("sha256") or "")

    def parents_of(self, node: str) -> set[str]:
        return self.parents.get(node, set())

    def ancestors(self, node: str) -> set[str]:
        cached = self._ancestors.get(node)
        if cached is not None:
            return cached
        seen: set[str] = set()
        stack = list(self.parents_of(node))
        while stack:
            parent = stack.pop()
            if parent in seen:
                continue
            seen.add(parent)
            stack.extend(self.parents_of(parent))
        self._ancestors[node] = seen
        return seen

    def depth(self, node: str) -> int | None:
        if node in self._depth:
            return self._depth[node]
        seen = {node}
        queue: deque[tuple[str, int]] = deque([(node, 0)])
        found: int | None = None
        while queue:
            current, dist = queue.popleft()
            if current == "CL:0000000":
                found = dist
                break
            for parent in self.parents_of(current):
                if parent not in seen:
                    seen.add(parent)
                    queue.append((parent, dist + 1))
        self._depth[node] = found
        return found

    def up_dist(self, src: str, dst: str) -> tuple[int, list[str]] | None:
        """Shortest directed distance from src up is_a edges to dst, and one path."""
        queue: deque[tuple[str, list[str]]] = deque([(src, [src])])
        seen = {src}
        while queue:
            current, path = queue.popleft()
            if current == dst:
                return len(path) - 1, path
            for parent in sorted(self.parents_of(current)):
                if parent not in seen:
                    seen.add(parent)
                    queue.append((parent, path + [parent]))
        return None

    def lca(self, left: str, right: str) -> str | None:
        common = (self.ancestors(left) | {left}) & (self.ancestors(right) | {right})
        if not common:
            return None

        def key(node: str) -> tuple[int, str]:
            dist = self.depth(node)
            return (-1 if dist is None else -dist, node)

        return min(common, key=key)


def load_ontology(path: Path) -> CellOntology:
    file = Path(path)
    blob = file.read_bytes()
    data = json.loads(blob)
    if "parents" in data and "labels" in data:
        source = dict(data.get("source") or {})
        source.setdefault("sha256", hashlib.sha256(blob).hexdigest())
        parents = {node: set(items) for node, items in data["parents"].items()}
        return CellOntology(parents, data["labels"], source)
    if "edges" in data and "id_2_lbl" in data:
        parents, labels = _from_popv_snapshot(data)
        source = {
            "name": file.name,
            "sha256": hashlib.sha256(blob).hexdigest(),
            "edges": sum(len(items) for items in parents.values()),
        }
        return CellOntology(parents, labels, source)
    raise ValueError(f"{file} is not a Cell Ontology snapshot")


def _from_popv_snapshot(data: dict) -> tuple[dict[str, set[str]], dict[str, str]]:
    labels = {_norm_id(uri): label for uri, label in data["id_2_lbl"].items()}
    parents: dict[str, set[str]] = {}
    for edge in data["edges"]:
        if edge.get("pred") != "is_a":
            continue
        child = _norm_id(edge["sub"])
        parent = _norm_id(edge["obj"])
        parents.setdefault(child, set()).add(parent)
    used = set(parents)
    for items in parents.values():
        used.update(items)
    return parents, {node: labels[node] for node in used if node in labels}


def _norm_id(uri: str) -> str:
    token = uri.rstrip("/").rsplit("/", 1)[-1]
    if token.startswith("CL_"):
        return "CL:" + token[3:]
    return token


def _reject_cycles(parents: dict[str, set[str]], nodes: set[str]) -> None:
    indegree = {node: 0 for node in nodes}
    for child, items in parents.items():
        indegree.setdefault(child, 0)
        for parent in items:
            indegree[parent] = indegree.get(parent, 0) + 1
    queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    seen = 0
    while queue:
        node = queue.popleft()
        seen += 1
        for parent in parents.get(node, ()):
            indegree[parent] -= 1
            if indegree[parent] == 0:
                queue.append(parent)
    if seen == len(indegree):
        return
    remaining = {node for node, degree in indegree.items() if degree > 0}
    raise OntologyCycleError(_one_cycle(parents, remaining))


def _one_cycle(parents: dict[str, set[str]], remaining: set[str]) -> list[str]:
    for start in sorted(remaining):
        path: list[str] = []
        index: dict[str, int] = {}
        node = start
        while node in remaining:
            if node in index:
                return path[index[node] :] + [node]
            index[node] = len(path)
            path.append(node)
            parent = next(
                (item for item in sorted(parents.get(node, ())) if item in remaining), None
            )
            if parent is None:
                break
            node = parent
    return [sorted(remaining)[0]]
