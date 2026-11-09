# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Cluster workflow: marker, name, and ML results, then a judgment writes the title."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Collection
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse

from .annotate import annotate
from .arbitrate import DEFAULT_MIN_MARGIN
from .decide import (
    SYSTEM_ML,
    SYSTEM_NAME,
    Decision,
    MlResult,
    candidate_names,
    cluster_payload,
    detection_fractions,
    fetch_ml,
    fetch_name,
    hit_names,
    ml_user_from_hits,
    parse_decision,
    parse_ml,
    propose_hypotheses,
    rank_measurements,
    trial_support,
    wilcoxon_order,
)
from .evidence import NEGATIVE_PRESENT_PCT, cluster_mask, require_unique_var_names
from .hypotheses import Hypothesis
from .marker_plot import (
    MAX_MARKERS_PER_TYPE,
    MAX_PVALUE,
    MIN_LOGFC,
    MIN_PCT_IN,
    plot_marker_panels,
    select_marker_panels,
)
from .markers import hypotheses_from_names, marker_names, parents_for, tissue_pages
from .prepare import _timed, doublet_summary, format_doublets
from .votes import agreed_name, gene_vote, panel_vote, subcluster_vote

SUBCLUSTER_KEY = "annotune_sub"
LABEL_KEY = "annotune_label"
MERGED_KEY = "annotune_cluster"
MIN_SUBCLUSTER_CELLS = 40
MIN_AUTOLABEL_CELLS = 20
SUBCLUSTER_RESOLUTIONS = (0.6, 1.0, 1.4)


@dataclass(frozen=True)
class TaskEndpoint:
    """One adapter. ``ask`` overrides HTTP and local weights. ``system`` overrides the task prompt."""

    url: str | None = None
    adapter: str | None = None
    system: str = ""
    ask: Callable | None = None
    maxlen: int = 2048
    base: str = ""
    tokenizer: str | None = None
    server: str | None = None


@dataclass(frozen=True)
class ModelChain:
    name: TaskEndpoint
    ml: TaskEndpoint


def local_models(
    root: str | Path, *, url: str | None = None, tokenizer: str | None = None
) -> ModelChain:
    """Q6_K base and the name and ML adapters.

    ``url`` is the server root. The default is ``http://127.0.0.1:8080``.
    A server is reused only when its base and both adapters match.
    The tokenizer is the one installed with the package.
    """
    from .local_lora import packaged_tokenizer, service_url

    root = Path(root).expanduser().resolve()
    base = str(root / "base" / "Qwen3.5-4B-base.Q6_K.gguf")
    chosen = url or service_url()
    tokenizer = packaged_tokenizer() if tokenizer is None else tokenizer

    def endpoint(name: str, maxlen: int) -> TaskEndpoint:
        return TaskEndpoint(
            adapter=str(root / "lora" / name),
            base=base,
            maxlen=maxlen,
            tokenizer=tokenizer,
            server=chosen,
        )

    return ModelChain(
        name=endpoint("name/annotune-v9h-name-checkpoint-412.lora.f16.gguf", 2048),
        ml=endpoint("ml/annotune-v9e-ml-checkpoint-231.lora.f16.gguf", 8192),
    )


_HINTS = {
    "small_cluster": f"fewer than {MIN_AUTOLABEL_CELLS} cells",
    "model_failure": "model call failed",
}
_LINEAGE = {"exact", "ancestor", "descendant"}
_SOURCE_TITLES = (("marker", "marker"), ("name", "name"), ("ML", "ml"))


def ontology_raw_labels() -> frozenset[str] | None:
    """Names mapped to a CL id in any species. None if the map is absent."""
    _dag, label_map, _names = _lineage_resources()
    if label_map is None:
        return None
    return frozenset(entry.raw_label for entry in label_map.entries if entry.cl_id)


def _lineage_resources(species: str | None = None):
    """DAG, map, and the names the map sends to a CL id for ``species``.

    All three are None when either file is missing. A name mapped only for
    the other species is not in the set.
    """
    from .granularity import ontology_dir

    return _load_lineage(ontology_dir(), species)


@lru_cache(maxsize=8)
def _load_lineage(directory: Path, species: str | None):
    from .granularity.cl_dag import load_ontology
    from .granularity.ontology_map import load_map

    dag_path = directory / "cl_isa.json"
    map_path = directory / "ontology_map.jsonl"
    if not dag_path.is_file() or not map_path.is_file():
        return None, None, None
    label_map = load_map(map_path)
    names = set()
    for raw in {entry.raw_label for entry in label_map.entries}:
        found = label_map.lookup(raw, species)
        if found is not None and found.cl_id:
            names.add(raw)
    # The result is cached and shared, so callers get a set they cannot change.
    return load_ontology(dag_path), label_map, frozenset(names)


def settle_cluster(
    *,
    marker: str | None,
    name: str | None,
    ml: MlResult | None,
    added: list[str],
    cells: int | None = None,
    failed: bool = False,
    ontology_names: set[str] | None = None,
    species: str | None = None,
    ranking: list[dict] | None = None,
    margin: float | None = None,
    marker_labeled: bool = False,
    min_margin: float = DEFAULT_MIN_MARGIN,
    dag=None,
    label_map=None,
) -> dict[str, object]:
    """Write one name from marker, name, and ML.

    Exact, ancestor, and descendant Cell Ontology relations are one lineage.
    Siblings are not. All three calls on one lineage write the finer name when
    its marker score passes, with no hint. Two calls on one lineage write the
    ontology name on that lineage and name the dissenter in the hint. Three
    lineages still write the higher-scoring ontology name and add a conflict
    hint. A marker margin below ``min_margin`` does the same, unless name and
    ML agree on a descendant of the marker call: that descendant is written
    and the close-score hint stays. Platelet or megakaryocyte yields to a
    monocyte program that is also on. A cluster under 40 cells with two
    programs on is not named when marker, name, and ML do not all give the
    same name. When they do, that name is written and the hint lists every
    detected program from high score to low. A failed marker score with no
    ontology name writes nothing. Agreement does not pick the name.
    """
    ml_name = ml.names[0] if ml is not None and ml.status == "named" and ml.names else None
    proposals: list[dict[str, str]] = []
    if name:
        proposals.append({"source": "name", "name": name})
    if ml_name:
        proposals.append({"source": "ml", "name": ml_name})
    spoken = {"marker": marker, "name": name, "ml": ml_name}
    if species and dag is None:
        dag, label_map, loaded = _lineage_resources(species)
        if ontology_names is None:
            ontology_names = loaded
    label, other, close = _choose_label(
        spoken,
        ontology_names,
        species=species,
        ranking=ranking,
        margin=margin,
        marker_labeled=marker_labeled,
        min_margin=min_margin,
        dag=dag,
        label_map=label_map,
    )
    leukocyte = _monocyte_over_platelet(label, ranking, species, dag, label_map)
    if leukocyte:
        if label and label != leukocyte:
            other = [f"marker {label}", *[item for item in other if item != f"marker {label}"]]
        label = leukocyte
    mixed = (
        cells is not None
        and cells < MIN_SUBCLUSTER_CELLS
        and _mixed_programs(ranking, species, dag, label_map)
    )
    unanimous = bool(marker and name and ml_name and marker == name == ml_name)
    if mixed and not unanimous:
        label = None
    model_failed = failed or (ml is not None and ml.status == "failed")
    hint = None
    if mixed and label is None:
        status, reason = "review", "mixed_programs"
        hint = _program_hint(ranking, species, dag, label_map)
    elif label is None:
        if model_failed:
            status, reason = "review", "model_failure"
        elif cells is not None and cells < MIN_AUTOLABEL_CELLS:
            status, reason = "review", "small_cluster"
        else:
            status, reason = "escalate", "no_supported_name"
    else:
        caveats: list[str] = []
        if other or close:
            caveats.append("evidence_conflict")
        if cells is not None and cells < MIN_AUTOLABEL_CELLS:
            caveats.append("small_cluster")
        if model_failed:
            caveats.append("model_failure")
        status = "named"
        reason = "+".join(caveats) if caveats else "evidence_agreement"
        notes = []
        if other:
            notes.append("conflict: " + ", ".join(other))
        if close:
            notes.append("close marker scores")
        notes.extend(_HINTS[item] for item in caveats if item != "evidence_conflict")
        if mixed:
            notes.append(_program_hint(ranking, species, dag, label_map))
            if reason == "evidence_agreement":
                reason = "mixed_programs"
        if notes:
            hint = "; ".join(notes)
    return {
        "proposals": proposals,
        "added": list(added),
        "written_label": label,
        "hint": hint,
        "arbitration": status != "named",
        "final_status": status,
        "final_reason": reason,
    }


def _choose_label(
    spoken: dict[str, str | None],
    ontology_names: set[str] | None,
    *,
    species: str | None,
    ranking: list[dict] | None,
    margin: float | None,
    marker_labeled: bool,
    min_margin: float,
    dag,
    label_map,
) -> tuple[str | None, list[str], bool]:
    items = [(key, spoken[key]) for key in ("marker", "name", "ml") if spoken[key]]
    close = bool(ranking) and len(ranking) >= 2 and margin is not None and margin < min_margin
    if not items:
        return None, [], close

    def accepted(value: str) -> bool:
        return ontology_names is None or value in ontology_names

    components = _components(items, species, dag, label_map)
    components.sort(key=len, reverse=True)
    agreed_model = spoken["name"] if spoken["name"] and spoken["name"] == spoken["ml"] else None
    marker_name = spoken["marker"]
    if (
        close
        and agreed_model
        and marker_name
        and agreed_model != marker_name
        and accepted(agreed_model)
        and _is_descendant(agreed_model, marker_name, species, dag, label_map)
    ):
        return agreed_model, _outside(components, agreed_model), close
    agreed = (
        components[0] if len(components) == 1 or len(components[0]) > len(components[1]) else None
    )
    pool = agreed if agreed is not None else items
    names = list(dict.fromkeys(value for _key, value in pool if accepted(value)))
    if not names:
        names = list(dict.fromkeys(value for _key, value in items if accepted(value)))
    if not names:
        if marker_labeled and spoken["marker"]:
            return spoken["marker"], _outside(components, spoken["marker"]), close
        return None, [], close
    scores = {value: _marker_score(value, ranking) for value in names}
    finest = _finest(names, species, dag, label_map)
    marker = spoken["marker"]
    if marker_labeled and marker in finest:
        chosen = marker
    elif any(score is not None for score in scores.values()):
        chosen = max(
            names,
            key=lambda value: (
                scores[value] is not None,
                scores[value] if scores[value] is not None else 0.0,
                value == marker,
                value,
            ),
        )
    elif len(names) == 1:
        chosen = names[0]
    else:
        return None, [], close
    return chosen, _outside(components, chosen), close


def _components(
    items: list[tuple[str, str]], species, dag, label_map
) -> list[list[tuple[str, str]]]:
    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            if _same_lineage(items[left][1], items[right][1], species, dag, label_map):
                parent[find(right)] = find(left)
    groups: dict[int, list[tuple[str, str]]] = {}
    for index, item in enumerate(items):
        groups.setdefault(find(index), []).append(item)
    return list(groups.values())


def _same_lineage(left: str, right: str, species, dag, label_map) -> bool:
    if left == right:
        return True
    if not species or dag is None or label_map is None:
        return False
    from .granularity.relations import relation

    found = relation(left, right, species, dag, label_map, formal=False)
    return found["relation"] in _LINEAGE


def _finest(names: list[str], species, dag, label_map) -> list[str]:
    kept: list[str] = []
    for name in names:
        if any(
            other != name and _is_descendant(other, name, species, dag, label_map)
            for other in names
        ):
            continue
        kept.append(name)
    return kept


def _is_descendant(finer: str, coarser: str, species, dag, label_map) -> bool:
    if not species or dag is None or label_map is None:
        return False
    from .granularity.relations import relation

    return (
        relation(finer, coarser, species, dag, label_map, formal=False)["relation"] == "descendant"
    )


def _marker_score(name: str, ranking: list[dict] | None) -> float | None:
    for row in ranking or []:
        if row.get("name") == name:
            return float(row["score"])
    return None


def _program_hold(record, cells, species, dag, label_map) -> dict[str, object] | None:
    """Apply the platelet and mixed-program rules when no model call settled the cluster."""
    if record.get("status") != "label" or dag is None or label_map is None:
        return None
    ranking = [dict(row) for row in record.get("ranking") or []]
    label = str(record["cell_type"]) if record.get("cell_type") else None
    if (
        cells is not None
        and cells < MIN_SUBCLUSTER_CELLS
        and _mixed_programs(ranking, species, dag, label_map)
    ):
        return {
            "written_label": None,
            "hint": _program_hint(ranking, species, dag, label_map),
            "final_status": "review",
            "final_reason": "mixed_programs",
            "arbitration": True,
        }
    leukocyte = _monocyte_over_platelet(label, ranking, species, dag, label_map)
    if leukocyte and leukocyte != label:
        return {
            "written_label": leukocyte,
            "hint": f"conflict: marker {label}",
            "final_status": "named",
            "final_reason": "evidence_conflict",
            "arbitration": True,
        }
    return None


def _monocyte_over_platelet(chosen, ranking, species, dag, label_map) -> str | None:
    """Platelet RNA on a monocyte program is an aggregate, not a platelet cluster."""
    if chosen not in {"platelet", "megakaryocyte"} or dag is None or label_map is None:
        return None
    best_name = None
    best_score = None
    for row in ranking or []:
        if not row.get("program_on"):
            continue
        name = str(row["name"])
        if not _same_lineage(name, "monocyte", species, dag, label_map):
            continue
        score = float(row["score"])
        if best_name is None or score > best_score:
            best_name = name
            best_score = score
    return best_name


def _detected_programs(ranking, species, dag, label_map) -> list[tuple[str, float]]:
    """Ontology programs whose positive panel is on, highest score first."""
    if dag is None or label_map is None:
        return []
    found: list[tuple[str, float]] = []
    seen: set[str] = set()
    for row in ranking or []:
        name = row.get("name")
        if not name or not row.get("program_on") or name in seen:
            continue
        entry = label_map.lookup(str(name), species, formal=False)
        if entry is None or not entry.cl_id:
            continue
        seen.add(str(name))
        found.append((str(name), float(row["score"])))
    found.sort(key=lambda item: (-item[1], item[0]))
    return found


def _needs_query(record: dict) -> bool:
    """Unnamed clusters and named clusters that still carry a question."""
    return not record.get("written_label") or bool(record.get("hint"))


def _print_queries(records: list[dict]) -> None:
    rows = [record for record in records if record.get("query")]
    if not rows:
        return
    print("Queries", flush=True)
    for record in rows:
        label = record.get("written_label") or "no label"
        hint = record.get("hint") or record.get("final_reason") or ""
        print(f"cluster {record['cluster']}  {label}  {hint}", flush=True)
        print(record["query"], flush=True)
        print(flush=True)


def _attach_queries(adata, cluster_key, records, hypotheses, species, tissue) -> None:
    """Pasteable gene features for unnamed clusters and clusters with a question."""
    pending = [record for record in records if _needs_query(record)]
    if not pending:
        return
    if adata.obs[cluster_key].nunique() >= 2:
        try:
            wilcoxon_order(adata, cluster_key)
        except ValueError:
            return
    dag = label_map = None
    if species:
        dag, label_map, _known = _lineage_resources(species)
    by_name = {item.name: item for item in hypotheses}
    for record in pending:
        cluster = str(record["cluster"])
        cells = int((adata.obs[cluster_key].astype(str) == cluster).sum())
        programs = [
            name
            for name, _score in _detected_programs(record.get("ranking"), species, dag, label_map)
        ]
        text = _cluster_query(
            adata,
            cluster,
            species=species or "",
            tissue=tissue or "",
            cells=cells,
            programs=programs,
            by_name=by_name,
        )
        if text:
            record["query"] = text


def _cluster_query(
    adata,
    cluster: str,
    *,
    species: str,
    tissue: str,
    cells: int,
    programs: list[str],
    by_name: dict[str, Hypothesis],
) -> str | None:
    """Gene list a person can paste into a model and ask for a cell type."""
    result = adata.uns.get("rank_genes_groups")
    if not result or "names" not in result or "pts" not in result:
        return None
    key = str(cluster)
    try:
        names = [str(gene) for gene in result["names"][key]]
        changes = [float(value) for value in result["logfoldchanges"][key]]
        pct_in = result["pts"][key]
        pct_out = result["pts_rest"][key]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    change_of = dict(zip(names, changes, strict=True))

    def feature(gene: str) -> str | None:
        if gene not in change_of:
            return None
        inside = float(pct_in.loc[gene])
        outside = float(pct_out.loc[gene])
        return f"{gene} (in {inside:.2f}, out {outside:.2f}, logfc {change_of[gene]:.2f})"

    enriched: list[str] = []
    seen: set[str] = set()
    for gene, change in zip(names, changes, strict=True):
        if gene in seen or change <= 0:
            continue
        seen.add(gene)
        text = feature(gene)
        if text:
            enriched.append(text)
        if len(enriched) >= 10:
            break
    present: list[str] = []
    for name in programs:
        item = by_name.get(name)
        if item is None:
            continue
        for gene in (*item.positive, *item.support):
            if gene in seen or gene not in change_of:
                continue
            if float(pct_in.loc[gene]) < 0.5:
                continue
            seen.add(gene)
            text = feature(gene)
            if text:
                present.append(text)
            if len(present) >= 16:
                break
        if len(present) >= 16:
            break
    if not enriched and not present:
        return None
    subject = species or "unknown species"
    if tissue:
        subject = f"{subject}, {tissue}"
    lines = [f"{subject}, {cells} cells. What cell type is this cluster?"]
    if enriched:
        lines.append("Enriched versus other cells: " + "; ".join(enriched) + ".")
    if present:
        lines.append("Also detected: " + "; ".join(present) + ".")
    return "\n".join(lines)


def _program_hint(ranking, species, dag, label_map) -> str:
    names = [name for name, _score in _detected_programs(ranking, species, dag, label_map)]
    return "cell programs: " + ", ".join(names)


def _mixed_programs(ranking, species, dag, label_map) -> bool:
    """Two ontology lineages each have a detected positive program."""
    found = _detected_programs(ranking, species, dag, label_map)
    if len(found) < 2:
        return False
    groups = _components([("marker", name) for name, _score in found], species, dag, label_map)
    return len(groups) >= 2


def _outside(components: list[list[tuple[str, str]]], chosen: str) -> list[str]:
    home = next((group for group in components if any(name == chosen for _key, name in group)), [])
    home_keys = {key for key, _name in home}
    spoken = {key: name for group in components for key, name in group}
    return [
        f"{title} {spoken[key]}"
        for title, key in _SOURCE_TITLES
        if key in spoken and key not in home_keys
    ]


def dictionary_hypotheses(
    species: str,
    tissue: str | None = None,
    user: list[Hypothesis] | None = None,
) -> list[Hypothesis]:
    """Core and tissue dictionary, with user rows replacing the same name."""
    names = marker_names(species, tissue)
    merged = {item.name: item for item in hypotheses_from_names(names, species, tissue)}
    for item in user or []:
        merged[item.name] = item
    return list(merged.values())


def _in_matrix(
    hypotheses: list[Hypothesis], var_names: set[str]
) -> tuple[list[Hypothesis], list[dict[str, object]]]:
    """Keep a type whose positive markers are all measured.

    A negative gene absent from the matrix is omitted. Absence is not evidence
    for the type, and it does not drop the type. A missing positive marker
    still drops the type. Each dropped row names those genes.
    """
    kept: list[Hypothesis] = []
    dropped: list[dict[str, object]] = []
    for item in hypotheses:
        missing = [gene for gene in item.positive if gene not in var_names]
        if missing:
            dropped.append(
                {"name": item.name, "reason": "missing positive markers", "missing": missing}
            )
            continue
        negatives = tuple(gene for gene in item.negative if gene in var_names)
        support = tuple(gene for gene in item.support if gene in var_names)
        if negatives == item.negative and support == item.support:
            kept.append(item)
        else:
            kept.append(Hypothesis(item.name, item.positive, negatives, support))
    return kept, dropped


def _variable_genes(part: AnnData) -> int:
    matrix = part.X
    if sparse.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        second = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
        variance = second - mean**2
    else:
        variance = np.asarray(matrix).var(axis=0)
    return int(np.count_nonzero(variance > 1e-8))


SILHOUETTE_SAMPLE = 2000


def _subcluster(part: AnnData) -> bool:
    if _variable_genes(part) < 2:
        return False
    n_comps = min(10, part.n_obs - 1, part.n_vars - 1)
    if n_comps < 2:
        return False
    sc.pp.pca(part, n_comps=n_comps, svd_solver="full")
    sc.pp.neighbors(part, n_pcs=n_comps)
    from sklearn.metrics import silhouette_score

    best_labels = None
    best_score = -1.0
    for resolution in SUBCLUSTER_RESOLUTIONS:
        sc.tl.leiden(
            part,
            resolution=resolution,
            key_added="annotune_sub_try",
            flavor="igraph",
            n_iterations=2,
            random_state=0,
        )
        labels = part.obs["annotune_sub_try"].astype(str)
        if int(labels.nunique()) < 2 or int(labels.nunique()) >= part.n_obs:
            continue
        if part.n_obs > SILHOUETTE_SAMPLE:
            groups = labels.groupby(labels).indices
            if len(groups) >= SILHOUETTE_SAMPLE:
                continue
            rng = np.random.default_rng(0)
            chosen = np.array([rng.choice(indices) for indices in groups.values()])
            remaining = np.setdiff1d(np.arange(part.n_obs), chosen, assume_unique=True)
            extra = rng.choice(remaining, SILHOUETTE_SAMPLE - len(chosen), replace=False)
            sample = np.concatenate((chosen, extra))
            score = float(silhouette_score(part.obsm["X_pca"][sample], labels.iloc[sample]))
        else:
            score = float(silhouette_score(part.obsm["X_pca"], labels))
        if score > best_score:
            best_score = score
            best_labels = labels.to_numpy()
    if best_labels is None:
        return False
    part.obs[SUBCLUSTER_KEY] = best_labels
    return True


def _cluster_order(cluster: str) -> tuple[int, object]:
    try:
        return (0, int(cluster))
    except ValueError:
        return (1, cluster)


def annotation_code(
    adata: AnnData,
    records: list[dict[str, object]],
    cluster_key: str,
    label_key: str,
    marker_dict: dict[str, list[str]],
    type_order: list[str],
) -> str:
    """Editable code for labels, type order, UMAP, and checked marker dotplot."""
    group_key = SUBCLUSTER_KEY if SUBCLUSTER_KEY in adata.obs else cluster_key
    units = adata.obs[group_key].astype(str)
    assigned: dict[str, str] = {}
    choices: dict[str, list[str]] = {}
    for unit in units.unique():
        labels = adata.obs.loc[units == unit, label_key]
        names = labels.dropna().astype(str).unique()
        if not labels.isna().any() and len(names) == 1:
            assigned[str(unit)] = str(names[0])
    for record in records:
        cluster = str(record["cluster"])
        if cluster in assigned:
            continue
        names: list[str] = []
        for item in list(record.get("choices") or []):
            if item not in names:
                names.append(str(item))
        for item in record.get("ranking") or []:
            candidate = str(item["name"])
            if candidate not in names:
                names.append(candidate)
            if len(names) >= 2:
                break
        if names:
            choices[cluster] = names[:2]
    order = sorted(assigned, key=_cluster_order)
    mapping = {cluster: assigned[cluster] for cluster in order}
    lines = [
        "import pandas as pd",
        "import scanpy as sc",
        "",
        "# cluster2celltype maps each cluster id to its name.",
        "# fine_order is the UMAP legend and the dotplot row order.",
        "# marker_dict is the dotplot. Gene order inside a list is the column order.",
        "cluster2celltype = {",
    ]
    for cluster, name in mapping.items():
        lines.append(f"    {cluster!r}: {name!r},")
    lines.append("}")
    if choices:
        lines.extend(["", "# unresolved clusters", "cluster_choices = {"])
        for cluster in sorted(choices, key=_cluster_order):
            lines.append(f"    {cluster!r}: {choices[cluster]!r},")
        lines.append("}")
    lines.extend(
        [
            "",
            "# Change this order to control the UMAP legend and dotplot rows.",
            "fine_order = [",
        ]
    )
    for name in type_order:
        lines.append(f"    {name!r},")
    lines.extend(
        [
            "]",
            "",
            f"assert {group_key!r} in adata.obs, 'run this on the annotated h5ad written by annotune'",
            f"adata.obs['celltype'] = adata.obs[{group_key!r}].astype(str).map(cluster2celltype)",
            "adata.obs['celltype'] = pd.Categorical(",
            "    adata.obs['celltype'], categories=fine_order, ordered=True",
            ")",
            "print(adata.obs['celltype'].isna().sum(), 'cells without a label')",
            "print(adata.obs['celltype'].value_counts())",
            "if 'X_umap' in adata.obsm:",
            "    sc.pl.umap(adata, color='celltype', frameon=False)",
            "",
            "# Cited dictionary markers. Types flagged no_marker_passed show their best measured genes.",
            "# A shared gene is kept only when that type would otherwise have no marker.",
            "# Edit these lists to customize the figure; marker_validation in the report has the measurements.",
            "marker_dict = {",
        ]
    )
    for name in type_order:
        if name in marker_dict:
            lines.append(f"    {name!r}: {marker_dict[name]!r},")
    lines.append("}")
    if marker_dict:
        lines.extend(
            [
                "# Keep marker columns in the same order as the cell type rows.",
                "ordered_marker_dict = {name: marker_dict[name] for name in fine_order if name in marker_dict}",
                "sc.pl.dotplot(",
                "    adata[adata.obs['celltype'].notna()].copy(), ordered_marker_dict,",
                "    groupby='celltype', categories_order=fine_order,",
                "    dendrogram=False, use_raw=False, standard_scale='var',",
                ")",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def run(
    adata: AnnData,
    cluster_key: str,
    *,
    species: str | None = None,
    tissue: str | None = None,
    hypotheses: list[Hypothesis] | None = None,
    user_hypotheses: list[Hypothesis] | None = None,
    min_margin: float = DEFAULT_MIN_MARGIN,
    ranker: object | None = None,
    conformal_q: float | None = None,
    catalog: list[Hypothesis] | None = None,
    models: ModelChain | None = None,
    stage: str = "",
    source: str | None = None,
    donor: str | None = None,
    model_clusters: Collection[str] | None = None,
    subcluster: bool = True,
    min_subcluster_cells: int = MIN_SUBCLUSTER_CELLS,
    label_key: str = LABEL_KEY,
    dotplot_path: Path | str | None = None,
    show: bool = False,
) -> dict[str, object]:
    """Collect the marker, name, and ML results, then write the title.

    The marker score, the name model, and the ML model each return a result.
    A judgment reads the three results and writes the title. The title is a
    dictionary name. A cluster can stay unnamed. The name and ML models run
    when ``models`` is set. The name model copies a cell-type name from the
    marker dictionary. The ML model is trained on machine-learning measurements
    of the cluster, differential expression and marker-panel statistics, and
    copies a name from those candidates. Name and ML each receive their own
    payload. Clear clusters are written to ``label_key``. Uncertain clusters
    are subclustered once. Each step prints when it starts and how many seconds
    it took. A cluster with no written name, or a written name that still has a
    hint, prints a gene query that can be pasted into another model. A local
    server started for the adapters stays loaded after this call.

    With ``models``, the name and ML models are asked for every cluster, or
    only those in ``model_clusters``. That includes clusters the marker score
    already named. The judgment reads all three results and can write a
    different title.

    ``adata`` is changed in place. This writes ``obs`` columns ``label_key``,
    ``annotune_cluster``, ``annotune_final_status`` and ``annotune_hint``, and
    ``annotune_sub`` when a cluster is split. An ``annotune_sub`` column from
    an earlier run is removed first. It also sets ``uns['log1p']`` when the
    matrix passes the log1p check, and ``uns['rank_genes_groups']`` when it
    runs Wilcoxon.
    """
    if hypotheses is None:
        if species is None:
            raise ValueError("species is required when hypotheses are omitted")
        hypotheses = dictionary_hypotheses(species, tissue, user_hypotheses)
    elif user_hypotheses:
        merged = {item.name: item for item in hypotheses}
        for item in user_hypotheses:
            merged[item.name] = item
        hypotheses = list(merged.values())
    require_unique_var_names(adata)
    if SUBCLUSTER_KEY in adata.obs.columns:
        del adata.obs[SUBCLUSTER_KEY]
    kept, dropped = _in_matrix(hypotheses, set(map(str, adata.var_names)))
    if len(kept) < 1:
        raise ValueError("no hypothesis has its positive markers in var_names")
    pool = list(catalog) if catalog is not None else list(kept)
    started = time.perf_counter()
    with _timed("run  score markers") as note:
        records = _annotate(adata, cluster_key, kept, min_margin, ranker, conformal_q)
        labeled = sum(record["status"] == "label" for record in records)
        note["detail"] = f"{labeled} of {len(records)} clusters labeled"
    used = {item.name for item in kept}
    extras: list[Hypothesis] = []
    seen = set(used)
    for record in records:
        if record["status"] == "label":
            continue
        for item in propose_hypotheses(pool, list(record["evidence"]["de_not_in_markers"]), seen):
            if item.name not in seen:
                extras.append(item)
                seen.add(item.name)
    extras, _ = _in_matrix(extras, set(map(str, adata.var_names)))
    if extras:
        kept = [*kept, *extras]
        with _timed("run  score added markers") as note:
            records = _annotate(adata, cluster_key, kept, min_margin, ranker, conformal_q)
            note["detail"] = f"{len(extras)} added"
    children: list[dict[str, object]] = []
    if subcluster:
        with _timed("run  subcluster") as note:
            split_any = False
            for record in records:
                if record["status"] == "label":
                    continue
                mask = cluster_mask(adata, cluster_key, str(record["cluster"]))
                if int(mask.sum()) < min_subcluster_cells:
                    continue
                part = adata[mask].copy()
                if not _subcluster(part):
                    continue
                child_records = _annotate(
                    part, SUBCLUSTER_KEY, kept, min_margin, ranker, conformal_q
                )
                for child in child_records:
                    child["parent"] = str(record["cluster"])
                    child["cluster"] = f"{record['cluster']}.{child['cluster']}"
                if not split_any:
                    adata.obs[SUBCLUSTER_KEY] = adata.obs[cluster_key].astype(str)
                    split_any = True
                parent_labels = adata.obs.loc[mask, cluster_key].astype(str)
                child_labels = part.obs[SUBCLUSTER_KEY].astype(str)
                adata.obs.loc[mask, SUBCLUSTER_KEY] = [
                    f"{parent}.{child}"
                    for parent, child in zip(parent_labels, child_labels, strict=True)
                ]
                children.append({"parent": str(record["cluster"]), "records": child_records})
            note["detail"] = f"{len(children)} clusters"
    by_parent = {str(item["parent"]): item["records"] for item in children}
    resolved = _resolved_labels(adata, cluster_key, records, by_parent)
    with _timed("run  Wilcoxon"):
        _apply_votes(adata, cluster_key, records, by_parent, kept, resolved, species, tissue)
    arbitration: list[str] = []
    added_hypotheses: list[Hypothesis] = []
    if models is not None:
        with _timed("run  load models"):
            _open_local(models)
        added_hypotheses = _run_models(
            adata,
            cluster_key,
            models,
            records,
            kept,
            pool,
            resolved,
            arbitration,
            species=species or "",
            tissue=tissue or "",
            stage=stage,
            source=source,
            donor=_one_donor(adata, donor),
            model_clusters=model_clusters,
            min_margin=min_margin,
            ranker=ranker,
            conformal_q=conformal_q,
        )
    dag = label_map = None
    if species:
        dag, label_map, _known = _lineage_resources()
    for record in records:
        if "final_status" in record or dag is None:
            continue
        cluster = str(record["cluster"])
        current = resolved.get(cluster)
        if current is not None and set(current) != {cluster}:
            continue
        cells = int((adata.obs[cluster_key].astype(str) == cluster).sum())
        revised = _program_hold(record, cells, species, dag, label_map)
        if revised is None:
            continue
        record.update(revised)
        if revised["written_label"]:
            resolved[cluster] = {cluster: str(revised["written_label"])}
        else:
            resolved.pop(cluster, None)
    for record in records:
        if "final_status" in record:
            continue
        cluster = str(record["cluster"])
        mapping = resolved.get(cluster, {})
        record["final_status"] = "named" if mapping else "review"
        record["final_reason"] = (
            "subcluster_labels"
            if mapping and set(mapping) != {cluster}
            else "votes"
            if mapping and record.get("vote_label") == mapping.get(cluster)
            else "marker_rule"
            if mapping
            else "unresolved_evidence"
        )
        record["written_label"] = mapping.get(cluster)
        record["hint"] = None
    _apply_neighbor_titles(adata, cluster_key, records, resolved, species, dag, label_map)
    _attach_queries(adata, cluster_key, records, [*kept, *added_hypotheses], species, tissue)
    labels = _assign_labels(adata, cluster_key, resolved)
    written = bool(labels.notna().any())
    adata.obs[label_key] = labels.astype("category")
    status_by_cluster = {str(record["cluster"]): record["final_status"] for record in records}
    hint_by_cluster = {str(record["cluster"]): record.get("hint") for record in records}
    adata.obs["annotune_final_status"] = (
        adata.obs[cluster_key].astype(str).map(status_by_cluster).astype("category")
    )
    adata.obs["annotune_hint"] = (
        adata.obs[cluster_key].astype(str).map(hint_by_cluster).astype("string")
    )
    all_labeled = bool(written) and bool(labels.notna().all())
    with _timed("run  marker panels") as note:
        marker_dict, marker_validation, marker_status = select_marker_panels(
            adata, label_key, species, tissue
        )
        while _flag_labels_without_markers(records, resolved, marker_status, by_parent):
            labels = _assign_labels(adata, cluster_key, resolved)
            written = bool(labels.notna().any())
            adata.obs[label_key] = labels.astype("category")
            all_labeled = bool(written) and bool(labels.notna().all())
            marker_dict, marker_validation, marker_status = select_marker_panels(
                adata, label_key, species, tissue
            )
        # The loop above also flags names with no passing marker; publish both.
        status_by_cluster = {str(record["cluster"]): record["final_status"] for record in records}
        hint_by_cluster = {str(record["cluster"]): record.get("hint") for record in records}
        adata.obs["annotune_final_status"] = (
            adata.obs[cluster_key].astype(str).map(status_by_cluster).astype("category")
        )
        adata.obs["annotune_hint"] = (
            adata.obs[cluster_key].astype(str).map(hint_by_cluster).astype("string")
        )
        _attach_queries(adata, cluster_key, records, [*kept, *added_hypotheses], species, tissue)
        note["detail"] = f"{len(marker_dict)} types"
    type_order = list(dict.fromkeys(adata.obs[label_key].dropna().astype(str)))
    plot = (
        plot_marker_panels(adata, label_key, marker_dict, type_order, dotplot_path)
        if dotplot_path is not None
        else None
    )
    code = annotation_code(
        adata,
        [*records, *(child for item in children for child in item["records"])],
        cluster_key,
        label_key,
        marker_dict,
        type_order,
    )
    merged = merge_same_labels(adata, cluster_key, label_key)
    clusters = _cluster_rows(adata, cluster_key, records, species)
    report = {
        "records": records,
        "clusters": clusters,
        "review": _review_rows(clusters),
        "dropped_hypotheses": dropped,
        "subclusters": children,
        "queue": [
            str(record["cluster"])
            for record in records
            if labels[adata.obs[cluster_key].astype(str) == str(record["cluster"])].isna().any()
        ],
        "arbitration": arbitration,
        "merged": merged,
        "all_labeled": all_labeled,
        "written": written,
        "label_key": label_key if written else None,
        "dotplot": plot,
        "marker_dict": marker_dict,
        "marker_validation": marker_validation,
        "marker_status": marker_status,
        "cluster_key": cluster_key,
        "doublets": doublet_summary(adata.uns.get("annotune_prep")),
        "tissue": tissue,
        "tissues": tissue_pages(species) if species else [],
        "marker_thresholds": {
            "min_pct_in": MIN_PCT_IN,
            "min_logfc": MIN_LOGFC,
            "max_pvalue": MAX_PVALUE,
            "max_markers_per_type": MAX_MARKERS_PER_TYPE,
        },
        "annotation_code": code,
    }
    _print_queries(records)
    print(
        f"run  done  {time.perf_counter() - started:.1f}s  "
        f"{sum(bool(record.get('written_label')) for record in records)} of {len(records)} clusters named",
        flush=True,
    )
    if show:
        _show_annotation(adata, cluster_key, label_key, report)
        if marker_dict:
            plot_marker_panels(adata, label_key, marker_dict, type_order)
        print(code)
    return report


def _model_name(record: dict, step: str):
    for item in record.get("rounds") or []:
        if item.get("step") != step:
            continue
        if item.get("status") == "failed":
            return "failed"
        names = item.get("names") or []
        if names:
            return names[0]
        return item.get("action")
    return None


def _ontology_note(label_map, label: object, species: str | None) -> dict[str, object] | None:
    """Exact map row for a written label. Unreviewed ids stay pending."""
    if label is None or label_map is None:
        return None
    entry = label_map.lookup(str(label), species)
    if entry is None or not entry.cl_id:
        return {"status": "unmapped"}
    return {
        "cl_id": entry.cl_id,
        "cl_label": entry.cl_label,
        "match": entry.match,
        "status": entry.status,
        "pending": entry.status != "reviewed",
    }


def _cluster_rows(
    adata: AnnData, cluster_key: str, records: list[dict], species: str | None
) -> list[dict[str, object]]:
    """One row per cluster. Sources stay separate. Ontology is an exact map lookup."""
    label_map = _lineage_resources(species)[1] if species else None
    rows = []
    for record in sorted(records, key=lambda row: _cluster_order(str(row["cluster"]))):
        cluster = str(record["cluster"])
        label = record.get("written_label")
        rows.append(
            {
                "cluster": cluster,
                "cells": int((adata.obs[cluster_key].astype(str) == cluster).sum()),
                "marker": record.get("cell_type"),
                "name": _model_name(record, "name"),
                "ml": _model_name(record, "ml"),
                "label": label,
                "hint": record.get("hint"),
                "query": record.get("query"),
                "reason": record.get("final_reason"),
                "ontology": _ontology_note(label_map, label, species),
            }
        )
    return rows


def _review_rows(clusters: list[dict[str, object]]) -> list[dict[str, object]]:
    """Clusters to read first. The rank is not an accuracy score."""
    ranked = []
    for row in clusters:
        priority = _review_priority(row)
        if priority is None:
            continue
        ranked.append(
            {
                "cluster": row["cluster"],
                "label": row["label"],
                "reason": row["reason"],
                "hint": row["hint"],
                "query": row.get("query"),
                "ontology": row["ontology"],
                "priority": priority,
            }
        )
    ranked.sort(key=lambda row: (row["priority"], _cluster_order(str(row["cluster"]))))
    return ranked


def _review_priority(row: dict[str, object]) -> int | None:
    if not row.get("label"):
        return 0
    reason = str(row.get("reason") or "")
    hint = str(row.get("hint") or "")
    if "evidence_conflict" in reason or hint.startswith("conflict") or "close marker" in hint:
        return 1
    if "model_failure" in reason or "model call failed" in hint:
        return 2
    if "small_cluster" in reason or hint.startswith("fewer than"):
        return 3
    if hint:
        return 4
    return None


def _show_annotation(adata, cluster_key: str, label_key: str, report: dict) -> None:
    import pandas as pd

    shown = [
        {
            "cluster": row["cluster"],
            "cells": row["cells"],
            "marker": row["marker"],
            "name": row["name"],
            "ml": row["ml"],
            "label": row["label"],
            "hint": row["hint"],
            "reason": row["reason"],
        }
        for row in report["clusters"]
    ]
    table = pd.DataFrame(shown)
    try:
        from IPython.display import display

        display(table)
    except Exception:
        print(table.to_string(index=False))
    print(format_doublets(report["doublets"]))
    _print_dropped(report["dropped_hypotheses"])
    _print_review(report["review"])
    print("merged", report["merged"])
    sc.pl.umap(adata, color=label_key, legend_loc="right margin", frameon=False, show=True)
    sc.pl.umap(adata, color=MERGED_KEY, legend_loc="on data", title=MERGED_KEY, show=True)


def _print_dropped(dropped: list[dict[str, object]]) -> None:
    if not dropped:
        return
    print("Dropped hypotheses")
    for item in dropped:
        missing = ", ".join(str(gene) for gene in item["missing"])
        print(f"  {item['name']}: {item['reason']}: {missing}")


def _print_review(review: list[dict[str, object]]) -> None:
    print("Review")
    if not review:
        print("  none")
        return
    print("  Order is for manual review. It is not an accuracy score.")
    for item in review:
        label = item["label"] or "no label"
        ontology = item.get("ontology") or {}
        suffix = ""
        if ontology.get("cl_id"):
            state = "pending" if ontology.get("pending") else ontology.get("status")
            suffix = f" [{ontology['cl_id']}, {state}]"
        detail = item["hint"] or item["reason"]
        print(f"  {item['cluster']}  {label}{suffix}  {detail}")


def merge_same_labels(adata: AnnData, cluster_key: str, label_key: str) -> list[dict[str, object]]:
    """Give cells that share a written name one cluster id.

    The original ``cluster_key`` stays. ``annotune_cluster`` is the merged id.
    A cluster that still holds more than one written name is left as it is.
    Cells with no name keep their own id. Hints stay on the original records.
    """
    units = adata.obs[cluster_key].astype(str)
    if SUBCLUSTER_KEY in adata.obs:
        sub = adata.obs[SUBCLUSTER_KEY]
        sub_text = sub.astype(str)
        use_sub = sub.notna() & ~sub_text.isin(["", "nan", "None", "<NA>"])
        units = sub_text.where(use_sub, units)
    text = adata.obs[label_key].astype(str)
    named = (
        adata.obs[label_key].notna().to_numpy() & ~text.isin(["", "nan", "None", "<NA>"]).to_numpy()
    )
    unit = units.to_numpy()
    names = text.to_numpy()
    names_per_unit: dict[str, set[str]] = defaultdict(set)
    for uid, name, ok in zip(unit, names, named, strict=True):
        if ok:
            names_per_unit[str(uid)].add(str(name))
    by_name: dict[str, list[str]] = defaultdict(list)
    for uid, found in names_per_unit.items():
        if len(found) == 1:
            by_name[next(iter(found))].append(uid)
    chosen_for: dict[str, str] = {}
    merged: list[dict[str, object]] = []
    for name, members in by_name.items():
        ordered = sorted(members, key=_unit_key)
        chosen = ordered[0]
        for uid in ordered:
            chosen_for[uid] = chosen
        if len(ordered) > 1:
            merged.append({"name": name, "cluster": chosen, "members": ordered})
    new_ids = [
        chosen_for[str(uid)] if ok and str(uid) in chosen_for else str(uid)
        for uid, ok in zip(unit, named, strict=True)
    ]
    adata.obs[MERGED_KEY] = new_ids
    return merged


def _unit_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _one_donor(adata, donor: str | None) -> str | None:
    if donor is not None:
        return donor
    if "donor_id" in adata.obs.columns:
        values = adata.obs["donor_id"].astype(str).unique()
        if len(values) == 1:
            return str(values[0])
    return None


def _failure_text(exc: BaseException) -> str:
    if type(exc).__name__ == "OutOfMemoryError":
        return "OutOfMemoryError"
    return str(exc)


def _is_runtime_failure(exc: BaseException) -> bool:
    return isinstance(exc, (OSError, ValueError, RuntimeError))


def _open_local(models: ModelChain) -> bool:
    """Start one base when both tasks have a local GGUF adapter."""
    ends = (models.name, models.ml)
    if not any(item.adapter for item in ends):
        return False
    if any(item.ask is not None for item in ends):
        raise ValueError("local adapters cannot be combined with ask callbacks")
    if not all(item.adapter and not item.url for item in ends):
        raise ValueError("name and ml share one base; pass both GGUF adapters")
    from .local_lora import open_adapters, packaged_tokenizer

    base = models.name.base
    if models.ml.base != base:
        raise ValueError("name and ml share one GGUF base")
    if not str(base).endswith(".gguf"):
        raise ValueError("local adapters require a GGUF --base")
    gguf = base
    chosen = [item.tokenizer for item in ends if item.tokenizer]
    if len(set(chosen)) > 1:
        raise ValueError("name and ml share one tokenizer")
    tokenizer = chosen[0] if chosen else packaged_tokenizer()
    servers = {item.server for item in ends if item.server}
    if len(servers) > 1:
        raise ValueError("name and ml share one server")
    kwargs = {"url": next(iter(servers))} if servers else {}
    open_adapters(
        {"name": models.name.adapter, "ml": models.ml.adapter},
        gguf=gguf,
        tokenizer=tokenizer,
        **kwargs,
    )
    return True


def _run_models(
    adata,
    cluster_key: str,
    models: ModelChain,
    records: list[dict],
    kept: list[Hypothesis],
    pool: list[Hypothesis],
    resolved: dict,
    arbitration: list[str],
    *,
    species: str,
    tissue: str,
    stage: str,
    source: str | None,
    donor: str | None,
    model_clusters,
    min_margin: float,
    ranker,
    conformal_q,
) -> list[Hypothesis]:
    """Score model replies. A retest adds types for that cluster only."""
    chosen = None if model_clusters is None else {str(item) for item in model_clusters}
    found: list[Hypothesis] = []
    dag, label_map, known_names = _lineage_resources(species or None)
    tables = wilcoxon_order(adata, cluster_key)
    for index, record in enumerate(records):
        cluster = str(record["cluster"])
        if chosen is not None and cluster not in chosen:
            continue
        print(f"run  models  cluster {cluster} ...", flush=True)
        model_started = time.perf_counter()
        payload = cluster_payload(
            adata,
            cluster_key,
            cluster,
            kept,
            species=species,
            tissue=tissue,
            stage=stage,
            source=source,
            donor=donor,
            tables=tables,
        )
        if "marker" in payload:
            raise RuntimeError("model payload must not contain a marker field")
        rounds: list[dict[str, object]] = []
        decision, name_error = _invoke(
            models.name, payload, fetch_name, _parse_name, SYSTEM_NAME, "name"
        )
        name: str | None = None
        if name_error is not None:
            record["model_error"] = str(name_error)
            rounds.append({"step": "name", "status": "failed", "error": str(name_error)})
        else:
            name = decision.names[0] if decision is not None and decision.names else None
            rounds.append(
                {
                    "step": "name",
                    "action": None if decision is None else decision.action,
                    "names": [] if decision is None else list(decision.names),
                }
            )
        try:
            positive, measurements = rank_measurements(adata, cluster)
        except (KeyError, TypeError, ValueError) as exc:
            measurements = None
            ml_result, ml_error = None, exc
        else:
            ml_result, ml_error = _call_ml(models.ml, payload, kept, positive, measurements)
        if ml_error is not None:
            ml_result = MlResult("failed", error=_failure_text(ml_error))
        rounds.append(
            {
                "step": "ml",
                "status": ml_result.status,
                "names": list(ml_result.names),
                "error": ml_result.error,
            }
        )
        added: list[Hypothesis] = []
        if name is not None and record.get("status") != "label":
            proposal = Decision("retest", (name,))
            added = _retest_hypotheses(proposal, pool, set(kept_names(kept)))
            added, _ = _in_matrix(added, set(map(str, adata.var_names)))
            if added:
                found.extend(added)
                updated = _annotate(
                    adata,
                    cluster_key,
                    [*kept, *added],
                    min_margin,
                    ranker,
                    conformal_q,
                    clusters=[cluster],
                )[0]
                # The vote views were recorded on the record this one replaces.
                for field in ("votes", "vote_label", "escalate"):
                    if field in record:
                        updated[field] = record[field]
                records[index] = updated
                record = updated
                if record.get("status") == "label":
                    resolved[cluster] = {cluster: str(record["cell_type"])}
        ranking = [dict(row) for row in record.get("ranking") or []]
        top = str(ranking[0]["name"]) if ranking else None
        labeled = bool(
            record.get("status") == "label"
            and record.get("cell_type")
            and top == str(record["cell_type"])
        )
        if top is None and record.get("status") == "label" and record.get("cell_type"):
            top = str(record["cell_type"])
            labeled = True
        settled = settle_cluster(
            marker=top,
            name=name,
            ml=ml_result,
            added=[item.name for item in added],
            cells=int(cluster_mask(adata, cluster_key, cluster).sum()),
            failed=name_error is not None,
            ontology_names=known_names,
            species=species or None,
            ranking=ranking,
            margin=None if record.get("margin") is None else float(record["margin"]),
            marker_labeled=labeled,
            min_margin=min_margin,
            dag=dag,
            label_map=label_map,
        )
        if settled["written_label"]:
            resolved[cluster] = {cluster: str(settled["written_label"])}
        elif settled["final_reason"] == "mixed_programs":
            resolved.pop(cluster, None)
        else:
            mapping = resolved.get(cluster) or {}
            if mapping and set(mapping) != {cluster}:
                settled = {
                    **settled,
                    "written_label": None,
                    "hint": None,
                    "arbitration": False,
                    "final_status": "named",
                    "final_reason": "subcluster_labels",
                }
            else:
                resolved.pop(cluster, None)
        record["rounds"] = rounds
        record["proposals"] = settled["proposals"]
        record["added"] = settled["added"]
        record["written_label"] = settled["written_label"]
        record["hint"] = settled["hint"]
        record["arbitration"] = settled["arbitration"]
        record["final_status"] = settled["final_status"]
        record["final_reason"] = settled["final_reason"]
        record["ml_trial"] = [] if measurements is None else _trial_for(payload, kept, measurements)
        if settled["arbitration"]:
            arbitration.append(cluster)
        written = settled.get("written_label") or "no label"
        print(
            f"run  models  cluster {cluster}  {time.perf_counter() - model_started:.1f}s  {written}",
            flush=True,
        )
    return found


def _invoke(endpoint: TaskEndpoint, payload: dict, fetch, parse, system: str, task: str):
    """Ask one task. ``fetch(url, payload)`` serves a URL. ``parse(text, payload)`` reads local text."""
    try:
        if endpoint.ask is not None:
            return endpoint.ask(payload), None
        if endpoint.adapter:
            from .local_lora import complete

            text = complete(task, endpoint.system or system, payload, maxlen=endpoint.maxlen)
            return parse(text, payload), None
        if endpoint.url:
            return fetch(endpoint.url, payload), None
        raise ValueError("task needs a url, an adapter, or an ask function")
    except Exception as exc:
        if _is_runtime_failure(exc):
            return None, exc
        raise


def _flag_labels_without_markers(
    records: list[dict], resolved: dict, marker_status: dict, by_parent: dict
) -> bool:
    """Flag a written name when none of its cited markers passed.

    The name stays written and the record keeps a hint, so it is still queued
    for review. The marker call replaces the name only when that call is a
    different name that did pass. Returns True when a name was replaced.
    """
    blocked = {
        name
        for name, status in marker_status.items()
        if status == "no_marker_passed_expression_check"
    }
    if not blocked:
        return False
    changed = False
    for record in records:
        label = record.get("written_label")
        cluster = str(record["cluster"])
        mapping = resolved.get(cluster)
        if mapping and set(mapping) != {cluster}:
            failed = {unit for unit, name in mapping.items() if name in blocked}
            if not failed:
                continue
            for child in by_parent.get(cluster, []):
                if str(child["cluster"]) in failed:
                    child["final_reason"] = "no_marker_passed"
                    child["hint"] = "no marker passed"
            record["hint"] = "one or more subclusters have no marker that passed validation"
            continue
        if label not in blocked or mapping != {cluster: label}:
            continue
        marker = record.get("cell_type") if record.get("status") == "label" else None
        if marker and marker != label and marker not in blocked:
            changed = True
            record["written_label"] = marker
            record["final_status"] = "named"
            record["final_reason"] = "marker_rule"
            record["hint"] = None
            resolved[cluster] = {cluster: str(marker)}
            continue
        if record.get("final_reason") == "no_marker_passed":
            continue
        record["final_status"] = "named"
        record["final_reason"] = "no_marker_passed"
        hint = record.get("hint")
        record["hint"] = "no marker passed" if not hint else f"{hint}; no marker passed"
    return changed


def _parse_name(text: str, payload: dict):
    return parse_decision(text, hit_names(payload))


def _parse_ml(text: str, payload: dict):
    return parse_ml(text, candidate_names(payload))


def _call_ml(endpoint: TaskEndpoint, payload: dict, hypotheses, positive, measurements):
    user = ml_user_from_hits(payload, hypotheses, measurements, positive)
    if "marker" in user:
        raise RuntimeError("ml payload must not contain a marker field")
    return _invoke(endpoint, user, fetch_ml, _parse_ml, SYSTEM_ML, "ml")


def _trial_for(payload: dict, hypotheses, measurements: dict[str, dict]) -> list[dict]:
    by_name = {item.name: item for item in hypotheses}
    rows = []
    for hit in payload.get("dictionary_hits") or []:
        item = by_name.get(str(hit["name"]))
        if item is None:
            continue
        positive = [measurements[gene] for gene in item.positive if gene in measurements]
        rows.append(
            {
                "name": item.name,
                **trial_support(positive, list(item.negative), measurements),
            }
        )
    return rows


def _annotate(adata, cluster_key, hypotheses, min_margin, ranker, conformal_q, clusters=None):
    return annotate(
        adata,
        cluster_key,
        hypotheses,
        min_margin=min_margin,
        ranker=ranker,
        conformal_q=conformal_q,
        clusters=clusters,
    )


def kept_names(hypotheses: list[Hypothesis]) -> list[str]:
    return [item.name for item in hypotheses]


def _children_labeled(children: list[dict[str, object]]) -> bool:
    return bool(children) and all(row["status"] == "label" for row in children)


def _child_label_map(children: list[dict[str, object]]) -> dict[str, str]:
    return {str(row["cluster"]): str(row["cell_type"]) for row in children}


def _apply_votes(
    adata, cluster_key, records, by_parent, hypotheses, resolved, species, tissue
) -> None:
    """Write a label when three views agree. Record every vote either way."""
    pending = [record for record in records if str(record["cluster"]) not in resolved]
    if not pending:
        return
    parents = parents_for(species=species, tissue=tissue) if species else {}
    known_names = _lineage_resources(species or None)[2]
    logfc = _logfc_by_cluster(adata, cluster_key)
    labeled = {
        str(record["cluster"]): str(record["cell_type"])
        for record in records
        if record["status"] == "label" and record.get("cell_type")
    }
    for record in pending:
        cluster = str(record["cluster"])
        children = list(by_parent.get(cluster, []))
        votes = {
            "panel": panel_vote(record),
            "genes": gene_vote(hypotheses, logfc.get(cluster, {})),
            "subcluster": subcluster_vote(children),
            "neighbors": _neighbor_vote(adata, cluster_key, cluster, labeled),
        }
        record["votes"] = votes
        name = agreed_name(votes, parents)
        if name is None:
            record["escalate"] = "views_disagree"
            continue
        # A shared parent can be a coarse name the dictionary and ontology do not have.
        if known_names is not None and name not in known_names:
            record["escalate"] = "vote_name_not_in_ontology"
            continue
        resolved[cluster] = {cluster: name}
        record["vote_label"] = name


def _logfc_by_cluster(adata: AnnData, cluster_key: str) -> dict[str, dict[str, float]]:
    """Log fold change of every gene, including genes that are lower inside.

    The name payload reads this same Wilcoxon table. A gene detected in fewer
    than ``NEGATIVE_PRESENT_PCT`` of the cluster's cells is never reported as
    higher inside, because its fold change is unstable.
    """
    if adata.obs[cluster_key].nunique() < 2 or _variable_genes(adata) < 2:
        return {}
    try:
        wilcoxon_order(adata, cluster_key)
    except ValueError:
        return {}
    result = adata.uns.get("rank_genes_groups")
    if not result or "names" not in result:
        return {}
    tables: dict[str, dict[str, float]] = {}
    for cluster in adata.obs[cluster_key].astype(str).unique():
        key = str(cluster)
        try:
            names = [str(gene) for gene in result["names"][key]]
            changes = [float(value) for value in result["logfoldchanges"][key]]
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        detected = detection_fractions(result["pts"], key, names) >= NEGATIVE_PRESENT_PCT
        tables[key] = {
            gene: change if seen or change < 0 else 0.0
            for gene, change, seen in zip(names, changes, detected, strict=True)
        }
    return tables


def _apply_neighbor_titles(adata, cluster_key, records, resolved, species, dag, label_map) -> None:
    """Give an unclear cluster the title of one clear cluster it touches.

    A title is clear when the name and ML calls both wrote it, or when no model
    ran and the marker rule wrote it without a hint. An unclear cluster takes
    that title when the neighbor graph or the UMAP places it with exactly one
    such lineage. A title the model already wrote on that lineage stays. Two
    clear lineages do not choose a title; a one-sided name from another lineage
    is withdrawn and the clear clusters are named in the hint.
    """
    if "connectivities" not in adata.obsp and "X_umap" not in adata.obsm:
        return
    by_id = {str(record["cluster"]): record for record in records}
    clear = {
        cluster: str(record["written_label"])
        for cluster, record in by_id.items()
        if _title_is_clear(record)
    }
    if not clear:
        return
    linked = _linked_clusters(adata, cluster_key)
    for record in records:
        cluster = str(record["cluster"])
        if cluster in clear or record.get("final_reason") == "subcluster_labels":
            continue
        mapping = resolved.get(cluster)
        if mapping and set(mapping) != {cluster}:
            continue
        neighbors = []
        seen: set[str] = set()
        for other in sorted(linked.get(cluster, ()), key=_cluster_order):
            if other not in clear or other in seen:
                continue
            seen.add(other)
            neighbors.append((other, clear[other]))
        if neighbors:
            _title_from_clear_neighbors(
                record, cluster, neighbors, resolved, species, dag, label_map
            )


def _title_is_clear(record: dict) -> bool:
    label = record.get("written_label")
    if not label or record.get("final_status") != "named":
        return False
    if record.get("final_reason") == "subcluster_labels":
        return False
    name = _proposal_name(record, "name")
    ml = _proposal_name(record, "ml")
    if name == label and ml == label:
        return True
    return (
        not record.get("rounds")
        and not record.get("hint")
        and record.get("final_reason") in {"marker_rule", "evidence_agreement", "votes"}
    )


def _proposal_name(record: dict, step: str):
    value = _model_name(record, step)
    if value in (None, "escalate", "failed", "refuse"):
        return None
    return str(value)


def _title_from_clear_neighbors(record, cluster, neighbors, resolved, species, dag, label_map):
    labels = list(dict.fromkeys(label for _other, label in neighbors))
    groups = _label_groups(labels, species, dag, label_map)
    near = "near " + ", ".join(f"{other} {label}" for other, label in neighbors)
    written = record.get("written_label")
    written_fits = bool(written) and any(
        _same_lineage(str(written), label, species, dag, label_map) for label in labels
    )
    if len(groups) != 1:
        if written and not written_fits:
            _withdraw_title(record, cluster, near, resolved)
        else:
            _join_hint(record, near)
        return
    if written_fits:
        _join_hint(record, near)
        return
    chosen = _coarsest(groups[0], species, dag, label_map)
    if chosen is None:
        if written and not written_fits:
            _withdraw_title(record, cluster, near, resolved)
        else:
            _join_hint(record, near)
        return
    note = _other_calls(record, chosen)
    record["written_label"] = chosen
    record["final_status"] = "named"
    record["final_reason"] = "near_clear_cluster"
    record["hint"] = "; ".join(part for part in (note, near) if part)
    resolved[cluster] = {cluster: chosen}


def _withdraw_title(record, cluster, near, resolved) -> None:
    note = _other_calls(record, None)
    record["written_label"] = None
    record["final_status"] = "review"
    record["final_reason"] = "near_clear_cluster"
    record["hint"] = "; ".join(part for part in (note, near) if part)
    resolved.pop(cluster, None)


def _other_calls(record: dict, title: str | None) -> str:
    parts = []
    marker = record.get("cell_type") if record.get("status") == "label" else None
    if marker and marker != title:
        parts.append(f"marker {marker}")
    name = _proposal_name(record, "name")
    if name and name != title:
        parts.append(f"name {name}")
    ml = _proposal_name(record, "ml")
    if ml and ml != title:
        parts.append(f"ml {ml}")
    return ", ".join(parts)


def _join_hint(record: dict, extra: str) -> None:
    current = record.get("hint")
    record["hint"] = extra if not current else f"{current}; {extra}"


def _label_groups(names: list[str], species, dag, label_map) -> list[list[str]]:
    parent = list(range(len(names)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            if _same_lineage(names[left], names[right], species, dag, label_map):
                parent[find(right)] = find(left)
    groups: dict[int, list[str]] = {}
    for index, name in enumerate(names):
        groups.setdefault(find(index), []).append(name)
    return list(groups.values())


def _coarsest(names: list[str], species, dag, label_map) -> str | None:
    unique = list(dict.fromkeys(names))
    if len(unique) == 1:
        return unique[0]
    parents = [
        name
        for name in unique
        if all(
            other == name or _is_descendant(other, name, species, dag, label_map)
            for other in unique
        )
    ]
    if len(parents) == 1:
        return parents[0]
    return None


def _linked_clusters(adata: AnnData, cluster_key: str) -> dict[str, set[str]]:
    """Clusters joined by the neighbor graph or by touching on the UMAP."""
    labels = adata.obs[cluster_key].astype(str).to_numpy()
    linked = {str(cluster): set() for cluster in pd.unique(labels)}
    if "connectivities" in adata.obsp:
        _add_links(linked, _connectivity_ratios(labels, adata.obsp["connectivities"]))
    if "X_umap" in adata.obsm:
        adjacency = _umap_adjacency(np.asarray(adata.obsm["X_umap"], dtype=float))
        _add_links(linked, _connectivity_ratios(labels, adjacency))
    return linked


def _add_links(linked: dict[str, set[str]], ratios: dict[str, list[tuple[str, float]]]) -> None:
    for cluster, pairs in ratios.items():
        linked.setdefault(cluster, set()).update(other for other, _ratio in pairs)


def _connectivity_ratios(labels: np.ndarray, adjacency) -> dict[str, list[tuple[str, float]]]:
    """PAGA's observed-over-expected edge ratio for each pair of clusters.

    A pair is returned only when the ratio is at least 1, the same comparison
    scanpy uses before capping the value for display.
    """
    matrix = sparse.csr_matrix(adjacency)
    n = matrix.shape[0]
    empty: dict[str, list[tuple[str, float]]] = {}
    if n == 0 or n != len(labels) or matrix.shape[1] != n:
        return empty
    ones = matrix.copy()
    ones.data = np.ones(len(ones.data), dtype=float)
    ones.setdiag(0)
    ones.eliminate_zeros()
    clusters = [str(cluster) for cluster in pd.unique(labels)]
    index = {name: pos for pos, name in enumerate(clusters)}
    codes = np.array([index[str(label)] for label in labels])
    size = len(clusters)
    outgoing = np.zeros((size, size), dtype=float)
    internal = np.zeros(size, dtype=float)
    coo = ones.tocoo()
    for row, col in zip(coo.row, coo.col, strict=True):
        left = codes[row]
        right = codes[col]
        if left == right:
            internal[left] += 1.0
        else:
            outgoing[left, right] += 1.0
    counts = np.bincount(codes, minlength=size).astype(float)
    strength = internal + outgoing.sum(axis=1)
    between = outgoing + outgoing.T
    result = {name: [] for name in clusters}
    denom = max(n - 1, 1)
    for left in range(size):
        for right in range(left + 1, size):
            observed = between[left, right]
            if observed <= 0:
                continue
            expected = (strength[left] * counts[right] + strength[right] * counts[left]) / denom
            ratio = observed / expected if expected else 1.0
            if ratio >= 1:
                result[clusters[left]].append((clusters[right], float(ratio)))
                result[clusters[right]].append((clusters[left], float(ratio)))
    return result


def _umap_adjacency(coords: np.ndarray):
    """Cells within one typical UMAP neighbor-spacing of each other."""
    from scipy.spatial import cKDTree

    n = coords.shape[0]
    if n < 2:
        return sparse.csr_matrix((n, n))
    distances, _indexes = cKDTree(coords).query(coords, k=2)
    spacing = float(np.median(distances[:, 1]))
    if not np.isfinite(spacing) or spacing <= 0:
        return sparse.csr_matrix((n, n))
    rows, cols = cKDTree(coords).query_pairs(spacing, output_type="ndarray").T
    if len(rows) == 0:
        return sparse.csr_matrix((n, n))
    data = np.ones(len(rows) * 2, dtype=float)
    return sparse.csr_matrix(
        (data, (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(n, n),
    )


def _neighbor_vote(
    adata: AnnData, cluster_key: str, cluster: str, labeled: dict[str, str]
) -> str | None:
    if "connectivities" not in adata.obsp:
        return None
    labels = adata.obs[cluster_key].astype(str).to_numpy()
    inside = np.flatnonzero(labels == str(cluster))
    if len(inside) == 0:
        return None
    weights = np.asarray(adata.obsp["connectivities"][inside].sum(axis=0)).ravel()
    names = pd.Series(labels).map(labeled).to_numpy(dtype=object)
    usable = (weights > 0) & (labels != str(cluster)) & pd.notna(names)
    if not usable.any():
        return None
    totals = pd.Series(weights[usable]).groupby(names[usable]).sum()
    if totals.sum() <= 0:
        return None
    name = str(totals.idxmax())
    if totals.max() / totals.sum() < 0.15:
        return None
    return name


def _resolved_labels(adata, cluster_key, records, by_parent) -> dict[str, dict[str, str]]:
    resolved: dict[str, dict[str, str]] = {}
    for record in records:
        cluster = str(record["cluster"])
        if record["status"] == "label":
            resolved[cluster] = {cluster: str(record["cell_type"])}
            continue
        children = list(by_parent.get(cluster, []))
        if _children_labeled(children):
            resolved[cluster] = _child_label_map(children)
    return resolved


def _retest_hypotheses(decision, pool: list[Hypothesis], used: set[str]) -> list[Hypothesis]:
    by_name = {item.name: item for item in pool}
    added: list[Hypothesis] = []
    for name in decision.names:
        item = by_name.get(name)
        if item is not None and name not in used:
            added.append(item)
            used.add(name)
    for item in decision.markers:
        if item.name not in used:
            added.append(item)
            used.add(item.name)
    return added


def _replace_record(records: list[dict[str, object]], updated: dict[str, object]) -> None:
    for index, record in enumerate(records):
        if str(record["cluster"]) == str(updated["cluster"]):
            records[index] = updated
            return


def _assign_labels(adata, cluster_key, resolved: dict[str, dict[str, str]]):
    import pandas as pd

    labels = pd.Series(pd.NA, index=adata.obs_names, dtype="string")
    clusters = adata.obs[cluster_key].astype(str)
    sub = adata.obs[SUBCLUSTER_KEY].astype(str) if SUBCLUSTER_KEY in adata.obs.columns else None
    for cluster, mapping in resolved.items():
        mask = clusters == cluster
        if set(mapping) == {cluster}:
            labels.loc[mask] = mapping[cluster]
            continue
        if sub is None:
            continue
        labels.loc[mask] = sub.loc[mask].map(mapping)
    return labels
