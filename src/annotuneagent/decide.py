# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Name and ML results for one cluster.

The name model copies a cell-type name from the marker dictionary. The ML
model reads measured differential expression and candidate marker panels, and
copies a name from those candidates. Both return a result when a model chain
is supplied. The workflow judgment reads those results together with the
marker score and writes the title.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
import weakref
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .evidence import NEGATIVE_PRESENT_PCT
from .hypotheses import Hypothesis

# Each task has its own prompt.
SYSTEM_NAME = (
    "You name one cell cluster. dictionary_hits is not in rank order. "
    "negative lists genes that type should not express when they are high in this cluster. "
    "Reply with JSON only. "
    '{"action": "retest", "names": ["dictionary name"]} '
    "Name a type only when at least two of its markers are near the top of the gene list "
    "and its negative list is empty. If two types list the same genes and only one has an empty "
    "negative list, name that type only when it is the one supported near the top. "
    "Copy the name from dictionary_hits."
)

SYSTEM_DECIDE = SYSTEM_NAME

SYSTEM_ML = (
    "You review one cell cluster using measured differential expression and candidate marker panels. "
    "genes is the full gene list retained by the cluster cache. marker_measurements contains "
    "every positive and negative marker for the listed candidates that was measured; null means "
    "the cache has no measurement, not zero expression. The columns are gene, logfc, pct_in, pct_out. "
    "Each candidate lists all its positive and negative marker names and statistics in the order "
    "specified by candidate_stat_columns. Use expression direction, detection rates, competing "
    'candidates, species and tissue. Reply with JSON only: {"names":["candidate name"]} '
    'or {"action":"escalate"}. Copy a name from candidates; do not invent one.'
)

ML_MEASUREMENT_COLUMNS = ["gene", "logfc", "pct_in", "pct_out"]
ML_STAT_COLUMNS = [
    "n_positive",
    "n_positive_supported",
    "n_positive_downregulated",
    "n_positive_detected",
    "n_negative",
    "n_negative_high",
    "n_negative_unmeasured",
    "mean_positive_logfc",
]

_ACTIONS = frozenset({"adopt_subcluster", "retest", "refuse", "escalate"})

# A marker counts as support only above this log2 fold change.
SUPPORT_LOGFC = 0.5


@dataclass(frozen=True)
class Decision:
    action: str
    names: tuple[str, ...] = ()
    markers: tuple[Hypothesis, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "names": list(self.names),
            "markers": [
                {
                    "name": item.name,
                    "positive": list(item.positive),
                    "negative": list(item.negative),
                }
                for item in self.markers
            ],
        }


def _gene_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("marker genes must be a list")
    return tuple(str(gene) for gene in value)


def parse_decision(text: str, allowed: set[str] | None = None) -> Decision:
    """Parse a model reply. Names outside ``allowed`` are dropped.

    An empty ``allowed`` set drops every name. ``None`` keeps them all.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("decision is not JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("decision must be an object")
    action = payload.get("action")
    if action not in _ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    raw_names = payload.get("names") or []
    raw_markers = payload.get("markers") or []
    if not isinstance(raw_names, list) or not isinstance(raw_markers, list):
        raise ValueError("names and markers must be lists")
    names = tuple(str(name) for name in raw_names if allowed is None or str(name) in allowed)
    markers: list[Hypothesis] = []
    for item in raw_markers:
        if not isinstance(item, dict):
            raise ValueError("marker entry must be an object")
        positive = _gene_list(item.get("positive"))
        negative = _gene_list(item.get("negative"))
        if not positive:
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("marker entry needs a name")
        markers.append(Hypothesis(name, positive=positive, negative=negative))
    if action == "retest" and not names and not markers:
        # A reply that only names types outside the hit list abstains.
        if allowed is not None and raw_names:
            return Decision("escalate")
        raise ValueError("retest needs a dictionary name or markers")
    if action != "retest":
        names = ()
        markers = ()
    return Decision(action, names, tuple(markers))


@dataclass(frozen=True)
class MlResult:
    """ML adapter result. ``named`` copied a candidate. ``escalate`` abstains. ``failed`` is not a prediction."""

    status: str
    names: tuple[str, ...] = ()
    error: str | None = None


def parse_ml(text: str, allowed: set[str] | None = None) -> MlResult:
    """Parse an ML reply. A ``names`` list is success. ``action=escalate`` abstains.

    A reply that is not a names list or an escalate action is rejected.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("ml reply is not JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("ml reply must be an object")
    if payload.get("action") == "escalate":
        return MlResult("escalate")
    raw_names = payload.get("names")
    if not isinstance(raw_names, list) or not raw_names:
        raise ValueError("ml success needs a names list")
    names = tuple(str(name) for name in raw_names if allowed is None or str(name) in allowed)
    if not names:
        if allowed is not None:
            return MlResult("escalate")
        raise ValueError("ml names are not in the candidate list")
    return MlResult("named", names)


def propose_hypotheses(
    catalog: list[Hypothesis],
    enriched: list[str],
    used: set[str],
    *,
    limit: int = 5,
) -> list[Hypothesis]:
    """Dictionary types whose positive markers overlap undeclared enriched genes."""
    genes = set(enriched)
    ranked: list[tuple[int, str, Hypothesis]] = []
    for item in catalog:
        if item.name in used:
            continue
        overlap = len(set(item.positive) & genes)
        if overlap:
            ranked.append((overlap, item.name, item))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [item for _, _, item in ranked[:limit]]


# The weak reference prevents a later AnnData object from reusing a dead one's id.
_WILCOXON: dict[int, tuple[weakref.ReferenceType, tuple, dict, dict]] = {}


def _gene_name(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _buffer_digest(values) -> bytes:
    arr = np.asarray(values)
    digest = hashlib.blake2b(digest_size=8)
    digest.update(arr.dtype.str.encode())
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    if arr.size == 0 or arr.dtype.kind == "O":
        return digest.digest()
    flat = arr.reshape(-1)
    if not flat.flags.c_contiguous:
        flat = np.ascontiguousarray(flat)
    digest.update(memoryview(flat).cast("B"))
    return digest.digest()


def _matrix_token(matrix) -> tuple:
    """Content token so an in-place edit of ``X`` does not reuse an old table."""
    if sparse.issparse(matrix):
        return (
            "sparse",
            tuple(matrix.shape),
            str(matrix.format),
            int(matrix.nnz),
            _buffer_digest(matrix.data),
            _buffer_digest(matrix.indices),
            _buffer_digest(matrix.indptr),
        )
    values = np.asarray(matrix)
    return ("dense", tuple(values.shape), str(values.dtype), _buffer_digest(values))


def wilcoxon_order(adata, cluster_key: str) -> dict[str, list[tuple[str, float]]]:
    """Positive genes in ``rank_genes_groups`` score order.

    Writes ``uns['rank_genes_groups']``. Repeated calls reuse the table when
    the cluster labels and the values in ``X`` are unchanged.
    """
    import scanpy as sc

    labels = adata.obs[cluster_key].astype(str)
    counts = labels.value_counts()
    matrix = adata.X
    signature = (cluster_key, tuple(labels), id(matrix), _matrix_token(matrix))
    cache_key = id(adata)
    cached = _WILCOXON.get(cache_key)
    if cached is not None and cached[0]() is adata and cached[1] == signature:
        adata.uns["rank_genes_groups"] = cached[3]
        return cached[2]
    if adata.obs[cluster_key].nunique() < 2:
        raise ValueError("wilcoxon needs at least two clusters")
    import warnings

    import pandas as pd

    # Scanpy inserts one column per cluster into a single frame.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        sc.tl.rank_genes_groups(
            adata,
            groupby=cluster_key,
            reference="rest",
            method="wilcoxon",
            use_raw=False,
            pts=True,
        )
    result = adata.uns["rank_genes_groups"]
    tables: dict[str, list[tuple[str, float]]] = {}
    for cluster in counts.index:
        key = str(cluster)
        try:
            order = [_gene_name(gene) for gene in result["names"][key]]
            changes = [float(value) for value in result["logfoldchanges"][key]]
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        rows: list[tuple[str, float]] = []
        seen: set[str] = set()
        for gene, change in zip(order, changes, strict=True):
            if gene in seen or change <= 0:
                continue
            seen.add(gene)
            rows.append((gene, change))
        tables[key] = rows
    _WILCOXON[cache_key] = (
        weakref.ref(adata, lambda _ref: _WILCOXON.pop(cache_key, None)),
        signature,
        tables,
        result,
    )
    return tables


def _detection(adata, cluster: str) -> dict[str, float]:
    result = adata.uns.get("rank_genes_groups")
    if not result or "pts" not in result:
        return {}
    frame = result["pts"]
    if cluster not in frame.columns:
        return {}
    return {_gene_name(gene): float(frame.loc[gene, cluster]) for gene in frame.index}


def ranked_payload(
    order: list[tuple[str, float]],
    pct: dict[str, float],
    hypotheses: list[Hypothesis],
    *,
    species: str = "",
    tissue: str = "",
    stage: str = "",
    previous: dict[str, object] | None = None,
    source: str | None = None,
    donor: str | None = None,
    cluster: str | None = None,
) -> dict[str, object]:
    """Action and name user object from a Wilcoxon-ordered gene list."""
    by_gene = dict(order)
    genes: list[dict[str, object]] = [
        {"gene": gene, "logfc": round(change, 1)} for gene, change in order[:25]
    ]
    scored: list[tuple[float, str, list[str]]] = []
    for item in hypotheses:
        chosen: list[str] = []
        total = 0.0
        for gene in item.positive:
            value = by_gene.get(gene)
            # Wilcoxon logFC explodes when a handful of cells carry the gene.
            # The same detection cut as a negative marker keeps those genes
            # out of the names the model is allowed to copy.
            if value is None or value <= SUPPORT_LOGFC:
                continue
            if pct.get(gene, 0.0) < NEGATIVE_PRESENT_PCT:
                continue
            chosen.append(gene)
            total += value
        if chosen:
            scored.append((total, item.name, chosen, item))
    scored.sort(key=lambda row: (-row[0], row[1]))
    hits = []
    for _, name, chosen, item in scored[:10]:
        forbidden = [
            gene
            for gene in item.negative
            if by_gene.get(gene, 0.0) >= 1 and pct.get(gene, 0.0) >= 0.25
        ]
        hits.append({"name": name, "genes": chosen, "negative": forbidden})
    have = {item["gene"] for item in genes}
    for hit in hits:
        for gene in [*hit["genes"], *hit["negative"]]:
            if gene in have or gene not in by_gene:
                continue
            genes.append({"gene": gene, "logfc": round(by_gene[gene], 1)})
            have.add(gene)
    if source is not None and cluster is not None:
        random.Random(f"{source}|{donor}|{cluster}").shuffle(hits)
    payload: dict[str, object] = {
        "species": species,
        "tissue": tissue,
        "stage": stage,
        "genes": genes,
        "dictionary_hits": hits,
    }
    if previous is not None:
        payload["previous"] = previous
    return payload


def cluster_payload(
    adata,
    cluster_key: str,
    cluster: str,
    hypotheses: list[Hypothesis],
    *,
    species: str = "",
    tissue: str = "",
    stage: str = "",
    previous: dict[str, object] | None = None,
    source: str | None = None,
    donor: str | None = None,
    tables: dict[str, list[tuple[str, float]]] | None = None,
) -> dict[str, object]:
    """User message: top genes plus dictionary hits.

    Gene order is the Wilcoxon score order. Marker genes with logfc>0 that
    fall outside the top 25 are appended. A caller that builds many clusters
    passes the ``wilcoxon_order`` result as ``tables``, so the matrix is not
    checked again for every cluster.
    """
    if tables is None:
        tables = wilcoxon_order(adata, cluster_key)
    order = tables.get(str(cluster), [])
    return ranked_payload(
        order,
        _detection(adata, str(cluster)),
        hypotheses,
        species=species,
        tissue=tissue,
        stage=stage,
        previous=previous,
        source=source,
        donor=donor,
        cluster=str(cluster),
    )


def wilcoxon_payload(
    adata,
    cluster_key: str,
    cluster: str,
    hypotheses: list[Hypothesis],
    *,
    species: str = "",
    tissue: str = "",
    stage: str = "",
    previous: dict[str, object] | None = None,
) -> dict[str, object]:
    """Training-order payload. ``cluster_payload`` uses this ranking."""
    return cluster_payload(
        adata,
        cluster_key,
        cluster,
        hypotheses,
        species=species,
        tissue=tissue,
        stage=stage,
        previous=previous,
    )


def rank_measurements(adata, cluster: str) -> tuple[list[dict], dict[str, dict]]:
    """Positive genes in Wilcoxon order, plus a measurement for every ranked gene.

    ``logfc`` is scanpy ``logfoldchanges``. It is not replaced by a detection-aware score.
    """
    result = adata.uns.get("rank_genes_groups")
    if not result:
        raise ValueError("rank_genes_groups is missing")
    key = str(cluster)
    names = [_gene_name(gene) for gene in result["names"][key]]
    changes = [float(value) for value in result["logfoldchanges"][key]]
    pct_in = detection_fractions(result["pts"], key, names)
    pct_out = detection_fractions(result["pts_rest"], key, names)
    measurements: dict[str, dict] = {}
    positive: list[dict] = []
    seen: set[str] = set()
    for gene, change, inside, outside in zip(names, changes, pct_in, pct_out, strict=True):
        row = {"gene": gene, "logfc": change, "pct_in": float(inside), "pct_out": float(outside)}
        measurements[gene] = row
        if change > 0 and gene not in seen:
            seen.add(gene)
            positive.append(row)
    return positive, measurements


def detection_fractions(frame, cluster: str, names: list[str]) -> np.ndarray:
    """Detection fraction of each gene in ``names``. A gene or cluster the table lacks is 0."""
    if cluster not in frame.columns:
        return np.zeros(len(names))
    return frame[cluster].reindex(names).fillna(0.0).to_numpy(dtype=float)


def _rounded_measurement(gene: str, row: dict) -> dict:
    return {
        "gene": gene,
        "logfc": round(float(row["logfc"]), 3),
        "pct_in": round(float(row["pct_in"]), 3),
        "pct_out": round(float(row["pct_out"]), 3),
    }


# The ML message keeps this many leading positive genes, then every other
# positive dictionary marker.
CACHE_HEAD = 24


def retained_measurements(
    positive: list[dict],
    hypotheses: list[Hypothesis],
    measurements: dict[str, dict],
) -> list[dict]:
    """Gene rows kept for the ML message, rounded to 3 decimals."""
    rows: list[dict] = []
    have: set[str] = set()
    for row in positive:
        if float(row["logfc"]) <= 0:
            continue
        gene = str(row["gene"])
        if gene in have:
            continue
        rows.append(_rounded_measurement(gene, row))
        have.add(gene)
        if len(rows) >= CACHE_HEAD:
            break
    for item in hypotheses:
        for gene in item.positive:
            if gene in have:
                continue
            src = measurements.get(gene)
            if src is None or float(src["logfc"]) <= 0:
                continue
            rows.append(_rounded_measurement(gene, src))
            have.add(gene)
    return rows


def ml_user_from_hits(
    action_payload: dict[str, object],
    hypotheses: list[Hypothesis],
    measurements: dict[str, dict],
    positive: list[dict],
) -> dict[str, object]:
    """ML user message. No marker field.

    ``positive`` is the Wilcoxon-ordered positive gene list. The emitted gene
    list is the retained list, not that whole list. Candidate order follows
    ``dictionary_hits``. Statistics use the unrounded measurement of each
    positive and negative marker.
    """
    by_name = {item.name: item for item in hypotheses}
    genes = retained_measurements(positive, hypotheses, measurements)
    precise = {str(row["gene"]): row for row in genes}
    for item in hypotheses:
        for gene in (*item.positive, *item.negative):
            src = measurements.get(gene)
            if src is None:
                continue
            precise[gene] = {
                "gene": gene,
                "logfc": float(src["logfc"]),
                "pct_in": float(src["pct_in"]),
                "pct_out": float(src["pct_out"]),
            }
    panels = []
    for hit in action_payload.get("dictionary_hits") or []:
        item = by_name.get(str(hit["name"]))
        if item is None:
            continue
        measured = [gene for gene in item.positive if gene in measurements]
        panels.append((item.name, measured, list(item.negative)))
    return ml_user(
        str(action_payload.get("species") or ""),
        str(action_payload.get("tissue") or ""),
        str(action_payload.get("stage") or ""),
        genes,
        panels,
        precise,
    )


def payload_agreement(runtime: dict[str, object], wilcoxon: dict[str, object]) -> dict[str, object]:
    """Overlap of the two rankings. A mismatch is a contract gap, not a model error."""

    def ranked(payload: dict[str, object]) -> list[str]:
        return [str(item["gene"]) for item in list(payload["genes"])[:25]]

    def hits(payload: dict[str, object]) -> list[tuple[str, tuple[str, ...]]]:
        return [
            (str(item["name"]), tuple(str(gene) for gene in item["genes"]))
            for item in list(payload["dictionary_hits"])
        ]

    left = ranked(runtime)
    right = ranked(wilcoxon)
    left_hits = hits(runtime)
    right_hits = hits(wilcoxon)
    return {
        "top25_shared": len(set(left) & set(right)),
        "top25_runtime": len(left),
        "top25_wilcoxon": len(right),
        "same_top25_order": left == right,
        "hit_names_runtime": [name for name, _ in left_hits],
        "hit_names_wilcoxon": [name for name, _ in right_hits],
        "same_hit_names": [name for name, _ in left_hits] == [name for name, _ in right_hits],
        "same_hit_genes": left_hits == right_hits,
    }


def user_text(task: str, payload: dict | str) -> str:
    """Serialize one user message.

    The name task uses spaced JSON. ML uses compact JSON.
    A string is sent unchanged.
    """
    if isinstance(payload, str):
        return payload
    separators = (",", ":") if task == "ml" else (", ", ": ")
    return json.dumps(payload, ensure_ascii=False, separators=separators)


def post_chat(
    url: str,
    system: str,
    payload: dict | str,
    timeout: float = 60,
    *,
    task: str = "name",
) -> str:
    """POST one chat completion. ``system`` is the task prompt, not a shared default."""
    endpoint = url.rstrip("/")
    if not endpoint.endswith("/v1/chat/completions"):
        endpoint += "/v1/chat/completions"
    body = {
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text(task, payload)},
        ],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode())
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"chat reply has no message content: {str(data)[:200]}") from exc


def hit_names(payload: dict[str, object]) -> set[str]:
    return {str(item["name"]) for item in payload.get("dictionary_hits") or []}


def fetch_name(
    url: str,
    payload: dict[str, object],
    timeout: float | None = None,
    *,
    maxlen: int = 2048,
    tokenizer: str | None = None,
) -> Decision:
    """Name task on ``/completion``. One adapter is enabled for this request."""
    from .local_lora import complete_url

    text = complete_url(
        url,
        "name",
        SYSTEM_NAME,
        payload,
        maxlen=maxlen,
        timeout=timeout,
        tokenizer=tokenizer,
    )
    return parse_decision(text, hit_names(payload))


def candidate_names(payload: dict[str, object]) -> set[str]:
    names: set[str] = set()
    for item in payload.get("candidates") or []:
        if isinstance(item, list) and item:
            names.add(str(item[0]))
        elif isinstance(item, dict) and item.get("name"):
            names.add(str(item["name"]))
    return names


def fetch_ml(
    url: str,
    payload: dict[str, object],
    timeout: float | None = None,
    *,
    maxlen: int = 8192,
    tokenizer: str | None = None,
) -> MlResult:
    """ML task on ``/completion``. One adapter is enabled for this request."""
    from .local_lora import complete_url

    text = complete_url(
        url,
        "ml",
        SYSTEM_ML,
        payload,
        maxlen=maxlen,
        timeout=timeout,
        tokenizer=tokenizer,
    )
    return parse_ml(text, candidate_names(payload))


def _negative_high(row: dict | None) -> bool:
    return row is not None and float(row["logfc"]) >= 1 and float(row["pct_in"]) >= 0.25


def ml_stat_row(positive: list[dict], negative: list[str], measurements: dict[str, dict]) -> list:
    """Training stat columns. Raw logfc is unchanged, including exploded scanpy values."""
    values = [float(row["logfc"]) for row in positive]
    return [
        len(positive),
        sum(float(row["logfc"]) > SUPPORT_LOGFC for row in positive),
        sum(float(row["logfc"]) < 0 for row in positive),
        sum(float(row["pct_in"]) >= 0.05 for row in positive),
        len(negative),
        sum(_negative_high(measurements.get(gene)) for gene in negative),
        sum(gene not in measurements for gene in negative),
        (sum(values) / len(values)) if values else None,
    ]


def trial_support(positive: list[dict], negative: list[str], measurements: dict[str, dict]) -> dict:
    """Detection-aware support. This is not written into the ML user message."""
    detected = [
        row
        for row in positive
        if float(row["logfc"]) > SUPPORT_LOGFC and float(row["pct_in"]) >= 0.25
    ]
    blocked = sum(_negative_high(measurements.get(gene)) for gene in negative)
    mean = sum(float(row["logfc"]) for row in detected) / len(detected) if detected else None
    return {
        "n_positive_supported_detected": len(detected),
        "mean_detected_logfc": mean,
        "n_negative_high": blocked,
        "blocked_by_negative": blocked > 0,
    }


def _measured(row: dict | None, gene: str) -> list:
    if row is None:
        return [gene, None, None, None]
    return [gene, row["logfc"], row["pct_in"], row["pct_out"]]


def ml_user(
    species: str,
    tissue: str,
    stage: str,
    genes: list[dict],
    panels: list[tuple[str, list[str], list[str]]],
    measurements: dict[str, dict],
) -> dict[str, object]:
    """ML user object. ``panels`` is ``(name, positive genes, negative genes)``."""
    selected: set[str] = set()
    candidates = []
    for name, positive, negative in panels:
        selected.update(positive)
        selected.update(negative)
        rows = [measurements[gene] for gene in positive if gene in measurements]
        candidates.append(
            [name, list(positive), list(negative), ml_stat_row(rows, negative, measurements)]
        )
    return {
        "species": species,
        "tissue": tissue,
        "stage": stage,
        "measurement_columns": list(ML_MEASUREMENT_COLUMNS),
        "genes": [_measured(row, str(row["gene"])) for row in genes],
        "marker_measurements": [
            _measured(measurements.get(gene), gene) for gene in sorted(selected)
        ],
        "candidate_stat_columns": list(ML_STAT_COLUMNS),
        "candidates": candidates,
    }
