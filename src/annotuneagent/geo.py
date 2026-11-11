# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Read a GEO series into one AnnData."""

from __future__ import annotations

import gzip
import re
import shutil
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import anndata as ad
import scanpy as sc
from anndata import AnnData

_ACCESSION = re.compile(r"GSE\d+", re.IGNORECASE)
_TIME = re.compile(r"^(?:d|day|dpi)[\s_-]?(\d+)$", re.IGNORECASE)
_REPLICATE = re.compile(r"^(?:b|s|r|rep|sample|replicate)[\s_-]?\d+$", re.IGNORECASE)
_WORDS = {
    "ctrl": "control",
    "con": "control",
    "control": "control",
    "wt": "wild type",
    "ko": "knockout",
    "het": "heterozygous",
    "veh": "vehicle",
    "vehicle": "vehicle",
    "sham": "sham",
    "untreated": "untreated",
    "treated": "treated",
    "naive": "naive",
}
_GROUP_KEYS = ("treatment", "condition", "group", "time", "timepoint", "stimulus", "disease")
_COUNTS = {"matrix.mtx.gz": "mtx", "matrix.h5": "h5"}


@dataclass(frozen=True)
class GeoSample:
    """One sample from a GEO series record.

    ``title`` is the name shown on the series page. ``counts`` is ``mtx``,
    ``h5``, or empty when the sample has no count matrix.
    """

    accession: str
    title: str
    organism: str
    characteristics: Mapping[str, str]
    counts: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "characteristics", MappingProxyType(dict(self.characteristics)))


@dataclass(frozen=True)
class GeoSeries:
    """A GEO series record without its count matrices.

    Print it to see every sample, then pass it to :func:`read_geo`.
    """

    accession: str
    title: str
    organism: str
    summary: str
    samples: tuple[GeoSample, ...]

    def __str__(self) -> str:
        lines = [f"{self.accession}  {self.title}".rstrip(), self.organism]
        if self.summary:
            lines.append(self.summary)
        lines.append("")
        for sample in self.samples:
            kind = {"mtx": "10x mtx", "h5": "10x h5"}.get(sample.counts, "no count matrix")
            lines.append(f"{sample.title}  {sample.accession}  {kind}")
            if sample.organism and sample.organism != self.organism:
                lines.append(f"  organism: {sample.organism}")
            for key, value in sample.characteristics.items():
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return str(self)


def geo_info(accession: str, *, cache: str | Path | None = None) -> GeoSeries:
    """Download a GEO series record and return its samples.

    ``accession`` is a series id such as ``GSE341410`` or a GEO URL. This
    reads the series record only. Count matrices stay on the server until
    :func:`read_geo`. Files are kept under ``cache``, or
    ``~/.cache/annotuneagent/geo`` when ``cache`` is omitted.
    """
    series, samples, meta, _root = _open(accession, cache)
    return _view(series, samples, meta)


def read_geo(
    accession: str | GeoSeries,
    *,
    samples: Sequence[str] | str | None = None,
    groups: Mapping[str, str] | str | None = None,
    cache: str | Path | None = None,
) -> AnnData:
    """Download GEO samples and join their count matrices.

    Pass a series id, a GEO URL, or the :class:`GeoSeries` returned by
    :func:`geo_info`. Omit ``samples`` to download every sample. To download
    some of them, pass titles or GSM accessions from that record::

        info = geo_info("GSE341410")
        adata = read_geo(info, samples=["B1", "B3"], groups="B1: control B3: Day7")

    ``samples`` may be a list or a string. A comma separates names. One title
    that contains spaces stays whole; several one-word titles can be separated
    by spaces. ``groups`` is the condition name for each sample you rename. A
    dict or a string both work, and the text is stored as written. A group key
    may be the full GEO title. ``obs["sample"]``
    stays the GEO title. ``obs["group"]`` is your name, or a condition read
    from the record when one differs between samples, or the title otherwise.
    A file already in the cache is not downloaded again.
    """
    series, records, meta, root = _open(accession, cache)
    chosen = _select(series, records, samples)
    labels, explained = _infer_groups(records)
    override = _parse_groups(groups, records)
    _check_groups(series, records, override)
    if not explained and not override:
        names = ", ".join(item.title for item in chosen)
        print(
            f"{series}: {names} share one recorded condition. Pass groups to name them.",
            flush=True,
        )
    labels = _apply_groups(chosen, labels, override)
    pieces: dict[str, AnnData] = {}
    for item in chosen:
        folder = _materialize(item, root / series)
        piece = _unique(_read_folder(folder))
        piece.obs["geo_accession"] = item.gsm
        for key, value in item.chars.items():
            column = _column(key)
            if column in ("sample", "group", "geo_accession"):
                column = f"geo_{column}"
            piece.obs[column] = value
        print(
            f"{item.title}: {piece.shape}  {item.gsm}  group {labels[item.title]}",
            flush=True,
        )
        pieces[item.title] = piece
    merged = ad.concat(pieces, label="sample", index_unique="-", join="outer", merge="same")
    merged.obs["sample"] = merged.obs["sample"].astype("category")
    merged.obs["group"] = merged.obs["sample"].astype(str).map(labels).astype("category")
    merged.uns["geo"] = {"accession": series, **meta, "groups": labels}
    return _unique(merged)


def _cache(cache: str | Path | None) -> Path:
    root = Path.home() / ".cache" / "annotuneagent" / "geo" if cache is None else Path(cache)
    return root.expanduser().resolve()


def _open(
    accession: str | GeoSeries, cache: str | Path | None
) -> tuple[str, list[_Sample], dict[str, str], Path]:
    text = accession.accession if isinstance(accession, GeoSeries) else accession
    series = _accession(text)
    root = _cache(cache)
    samples, meta = _parse_soft(_fetch_soft(series, root))
    if not samples:
        raise ValueError(f"{series} has no samples")
    counts: dict[str, list[str]] = {}
    for item in samples:
        counts.setdefault(item.title, []).append(item.gsm)
    repeated = [f"{title} ({', '.join(gsms)})" for title, gsms in counts.items() if len(gsms) > 1]
    if repeated:
        raise ValueError(f"{series} uses one title for more than one sample: {'; '.join(repeated)}")
    return series, samples, meta, root


def _view(series: str, samples: list[_Sample], meta: dict[str, str]) -> GeoSeries:
    rows = tuple(
        GeoSample(
            accession=item.gsm,
            title=item.title,
            organism=item.organism or meta["organism"],
            characteristics=item.chars,
            counts=_counts_kind(item),
        )
        for item in samples
    )
    return GeoSeries(
        accession=series,
        title=meta["title"],
        organism=meta["organism"],
        summary=meta["summary"],
        samples=rows,
    )


def _counts_kind(sample: _Sample) -> str:
    for url in sample.files:
        role = _role(_filename(url))
        if role in _COUNTS:
            return _COUNTS[role]
    return ""


def _accession(text: str) -> str:
    match = _ACCESSION.search(text)
    if match is None:
        raise ValueError("expected a GEO series accession, for example GSE341410")
    return match.group(0).upper()


def _column(key: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", key.strip().lower()).strip("_")


def _sample_names(samples: Sequence[str] | str, records: list[_Sample] | None = None) -> list[str]:
    if not isinstance(samples, str):
        return [str(name).strip() for name in samples if str(name).strip()]
    text = samples.replace("，", ",").replace("、", ",").strip()
    parts = [part.strip() for part in text.split(",") if part.strip()] if "," in text else [text]
    if not records:
        return [piece for part in parts for piece in re.split(r"\s+", part) if piece]
    known = {item.title for item in records} | {item.gsm.upper() for item in records}
    names: list[str] = []
    for part in parts:
        if part in known or part.upper() in known:
            names.append(part)
            continue
        pieces = [piece for piece in re.split(r"\s+", part) if piece]
        if pieces and all(piece in known or piece.upper() in known for piece in pieces):
            names.extend(pieces)
            continue
        names.append(part)
    return names


def _select(
    series: str, records: list[_Sample], samples: Sequence[str] | str | None
) -> list[_Sample]:
    if samples is None:
        return list(records)
    names = _sample_names(samples, records)
    if not names:
        raise ValueError("samples is empty")
    by_title = {item.title: item for item in records}
    by_gsm = {item.gsm.upper(): item for item in records}
    chosen: list[_Sample] = []
    unknown: list[str] = []
    for name in names:
        item = by_title.get(name) or by_gsm.get(name.upper())
        if item is None:
            unknown.append(name)
        else:
            chosen.append(item)
    if unknown:
        known = ", ".join(f"{item.title} ({item.gsm})" for item in records)
        raise ValueError(f"samples {unknown} are not in {series} ({known})")
    if len({item.gsm for item in chosen}) != len(chosen):
        raise ValueError("samples lists the same sample more than once")
    return chosen


def _parse_groups(
    groups: Mapping[str, str] | str | None, records: list[_Sample] | None = None
) -> dict[str, str]:
    if groups is None:
        return {}
    if not isinstance(groups, str):
        return {str(key): str(value) for key, value in groups.items()}
    text = groups.replace("，", " ").replace("：", ":")
    try:
        return _group_tokens(groups, text)
    except ValueError:
        if records is None:
            raise
    titled = _groups_by_title(text, records)
    if titled is None:
        raise ValueError(f"group {groups!r} needs a colon, for example B1: control")
    return titled


def _group_tokens(groups: str, text: str) -> dict[str, str]:
    parts = re.split(r"(\S+)\s*:\s*", text)
    if parts[0].strip() or len(parts) == 1:
        raise ValueError(f"group {groups!r} needs a colon, for example B1: control")
    found: dict[str, str] = {}
    for key, value in zip(parts[1::2], parts[2::2], strict=False):
        name = key.strip()
        label = value.strip()
        if not name or not label:
            raise ValueError(f"group {name or key!r} needs a name, for example B1: control")
        if name in found:
            raise ValueError(f"group {name} is repeated")
        found[name] = label
    return found


def _groups_by_title(text: str, records: list[_Sample]) -> dict[str, str] | None:
    """Read ``title: label`` when the title itself contains spaces."""
    keys = sorted(
        {item.title for item in records} | {item.gsm for item in records}, key=len, reverse=True
    )
    rest = text.strip()
    found: dict[str, str] = {}
    while rest:
        rest = rest.lstrip(" ,")
        if not rest:
            break
        key = next((name for name in keys if _starts_key(rest, name)), None)
        if key is None or key in found:
            return None
        rest = rest[len(key) :].lstrip()
        if not rest.startswith(":"):
            return None
        rest = rest[1:].lstrip()
        cuts = [index for name in keys for index in _key_at(rest, name)]
        cut = min(cuts) if cuts else len(rest)
        label = rest[:cut].strip(" ,")
        if not label:
            return None
        found[key] = label
        rest = rest[cut:]
    return found or None


def _starts_key(text: str, name: str) -> bool:
    if not text.startswith(name):
        return False
    return len(text) == len(name) or text[len(name)] in " \t:"


def _key_at(text: str, name: str) -> list[int]:
    found = []
    start = 0
    while True:
        index = text.find(name, start)
        if index < 0:
            return found
        before = index == 0 or text[index - 1].isspace()
        after = index + len(name) == len(text) or text[index + len(name)] in " \t:"
        if index > 0 and before and after:
            found.append(index)
        start = index + 1


def _check_groups(series: str, records: list[_Sample], override: Mapping[str, str]) -> None:
    titles = {item.title for item in records}
    gsms = {item.gsm.upper() for item in records}
    unknown = [key for key in override if key not in titles and key.upper() not in gsms]
    if unknown:
        known = ", ".join(item.title for item in records)
        raise ValueError(f"groups {unknown} are not samples of {series} ({known})")


def _apply_groups(
    chosen: list[_Sample], labels: Mapping[str, str], override: Mapping[str, str]
) -> dict[str, str]:
    by_gsm = {key.upper(): value for key, value in override.items()}
    applied: dict[str, str] = {}
    for item in chosen:
        if item.title in override:
            applied[item.title] = override[item.title]
        elif item.gsm.upper() in by_gsm:
            applied[item.title] = by_gsm[item.gsm.upper()]
        else:
            applied[item.title] = labels[item.title]
    return applied


def _infer_groups(samples: list[_Sample]) -> tuple[dict[str, str], bool]:
    """Return a group for each sample title, and whether a condition was found."""
    keys: list[str] = []
    for item in samples:
        for key in item.chars:
            if key not in keys:
                keys.append(key)
    varying = [key for key in keys if len({item.chars.get(key, "") for item in samples}) > 1]
    chosen = next((key for key in _GROUP_KEYS if key in varying), None)
    if chosen is None and len(varying) == 1:
        chosen = varying[0]
    if chosen is not None:
        return {item.title: _expand(item.chars.get(chosen, "")) for item in samples}, True
    from_title = {item.title: _expand_title(item.title) for item in samples}
    if any(_condition_tokens(item.title) for item in samples) and len(set(from_title.values())) > 1:
        return from_title, True
    return {item.title: item.title for item in samples}, False


def _condition_tokens(text: str) -> list[str]:
    return [token for token in _split(text) if not _REPLICATE.fullmatch(token)]


def _expand_title(text: str) -> str:
    whole = _TIME.fullmatch(text.strip())
    if whole:
        return f"day {int(whole.group(1))}"
    tokens = _condition_tokens(text)
    if not tokens:
        return text
    return ", ".join(_expand(token) for token in tokens)


def _expand(text: str) -> str:
    token = text.strip()
    timed = _TIME.fullmatch(token)
    if timed:
        return f"day {int(timed.group(1))}"
    return _WORDS.get(token.lower(), text)


def _split(text: str) -> list[str]:
    return [part for part in re.split(r"[\s_./+-]+", text.strip()) if part]


class _Sample:
    def __init__(self, gsm: str) -> None:
        self.gsm = gsm
        self.title = gsm
        self.chars: dict[str, str] = {}
        self.files: list[str] = []
        self.organism = ""


def _parse_soft(text: str) -> tuple[list[_Sample], dict[str, str]]:
    samples: list[_Sample] = []
    current: _Sample | None = None
    meta = {"title": "", "organism": "", "summary": ""}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("^SAMPLE"):
            current = _Sample(line.split("=", 1)[1].strip())
            samples.append(current)
            continue
        if not line.startswith("!"):
            continue
        key, _, value = line[1:].partition("=")
        key = key.strip()
        value = value.strip()
        if current is None:
            if key == "Series_title":
                meta["title"] = value
            elif key == "Series_summary":
                meta["summary"] = value
            elif key in ("Series_sample_organism", "Series_platform_organism"):
                meta["organism"] = meta["organism"] or value
            continue
        if key == "Sample_title":
            current.title = value
        elif key == "Sample_organism_ch1":
            current.organism = value
            meta["organism"] = meta["organism"] or value
        elif key.startswith("Sample_characteristics"):
            name, sep, detail = value.partition(":")
            if sep:
                current.chars[name.strip()] = detail.strip()
        elif key.startswith("Sample_supplementary_file"):
            current.files.append(value)
    return samples, meta


def _fetch_soft(accession: str, root: Path) -> str:
    number = accession[3:]
    url = (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/"
        f"GSE{number[:-3]}nnn/{accession}/soft/{accession}_family.soft.gz"
    )
    dest = root / accession / f"{accession}_family.soft.gz"
    _download(url, dest)
    raw = dest.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def _download(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "AnnoTuneAgent"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(dest)


def _https(url: str) -> str:
    if url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        return "https://" + url[len("ftp://") :]
    return url


def _filename(url: str) -> str:
    return url.rstrip("/").split("/")[-1].split("?")[0]


def _role(name: str) -> str | None:
    lower = name.lower()
    if lower.endswith("barcodes.tsv.gz") or lower.endswith("barcodes.tsv"):
        return "barcodes.tsv.gz"
    if (
        lower.endswith("features.tsv.gz")
        or lower.endswith("genes.tsv.gz")
        or lower.endswith("features.tsv")
        or lower.endswith("genes.tsv")
    ):
        return "features.tsv.gz"
    if lower.endswith("matrix.mtx.gz") or lower.endswith("matrix.mtx"):
        return "matrix.mtx.gz"
    if lower.endswith(".h5"):
        return "matrix.h5"
    return None


def _stored_name(filename: str) -> str | None:
    """Cache name for one 10x file. Compression stays as it was downloaded."""
    role = _role(filename)
    if role is None:
        return None
    if role == "matrix.h5":
        return "matrix.h5"
    compressed = filename.lower().endswith(".gz")
    plain = {
        "matrix.mtx.gz": "matrix.mtx",
        "barcodes.tsv.gz": "barcodes.tsv",
        "features.tsv.gz": "features.tsv",
    }[role]
    return role if compressed else plain


def _choose_h5(sample: _Sample) -> str | None:
    urls = [url for url in sample.files if _role(_filename(url)) == "matrix.h5"]
    if len(urls) <= 1:
        return urls[0] if urls else None
    filtered = [url for url in urls if "filtered" in _filename(url).lower()]
    if len(filtered) == 1:
        return filtered[0]
    names = ", ".join(_filename(url) for url in urls)
    raise ValueError(f"{sample.gsm} ({sample.title}) has more than one h5 matrix: {names}")


def _materialize(sample: _Sample, root: Path) -> Path:
    folder = root / sample.gsm
    folder.mkdir(parents=True, exist_ok=True)
    h5 = _choose_h5(sample)
    found = False
    for url in sample.files:
        name = _stored_name(_filename(url))
        if name is None or name == "matrix.h5":
            continue
        _download(_https(url), folder / name)
        found = True
    if h5 is not None:
        _download(_https(h5), folder / "matrix.h5")
        found = True
    if not found:
        raise FileNotFoundError(
            f"{sample.gsm} ({sample.title}) has no count matrix in its supplementary files"
        )
    return folder


def _read_folder(folder: Path) -> AnnData:
    h5 = folder / "matrix.h5"
    if (folder / "matrix.mtx.gz").is_file():
        return sc.read_10x_mtx(folder, var_names="gene_symbols", make_unique=True)
    if (folder / "matrix.mtx").is_file():
        return _read_plain_mtx(folder)
    if h5.is_file():
        return sc.read_10x_h5(h5)
    raise FileNotFoundError(f"{folder} has no matrix.mtx.gz or matrix.h5")


def _read_plain_mtx(folder: Path) -> AnnData:
    """Read an uncompressed 10x directory. Scanpy's mtx reader expects ``.gz``."""
    import numpy as np
    import pandas as pd
    from scipy import io, sparse

    matrix = io.mmread(folder / "matrix.mtx")
    if sparse.issparse(matrix):
        matrix = matrix.T.tocsr()
    else:
        matrix = np.asarray(matrix).T
    features = folder / "features.tsv"
    if not features.is_file():
        features = folder / "genes.tsv"
    table = pd.read_csv(features, sep="\t", header=None)
    barcodes = pd.read_csv(folder / "barcodes.tsv", sep="\t", header=None)
    names = table.iloc[:, 1 if table.shape[1] > 1 else 0].astype(str)
    names.name = None
    var = pd.DataFrame(index=names)
    if table.shape[1] > 1:
        var["gene_ids"] = table.iloc[:, 0].astype(str).to_numpy()
    if table.shape[1] > 2:
        var["feature_types"] = table.iloc[:, 2].astype(str).to_numpy()
    cells = barcodes.iloc[:, 0].astype(str)
    cells.name = None
    obs = pd.DataFrame(index=cells)
    data = AnnData(matrix, obs=obs, var=var)
    data.var_names_make_unique()
    return data


def _unique(adata: AnnData) -> AnnData:
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata
