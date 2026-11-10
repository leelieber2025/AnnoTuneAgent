# Changelog

All notable changes to this project are documented in this file.

## [0.1.2] - 2026-11-10

### Added

- `geo_info` reads a GEO series record: sample titles, accessions, and
  characteristics, without the count matrices. `read_geo` downloads the
  samples you name and joins them. `groups` sets `obs["group"]`. The GEO
  title stays in `obs["sample"]`.
- A tutorial runs [GSE266598](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE266598)
  from the series page through `prepare` and annotation.

### Fixed

- The CLI refuses to write the JSON report, the labeled h5ad, or the marker
  PDF onto an input file, or to use one path for two of those outputs.
- A T-cell subtype with no receptor gene detected in 25% of the cluster is
  left out of `dictionary_hits`, is not a vote, and is not written when it is
  the only name still in the judgment.
- `read_geo` keeps an uncompressed 10x file uncompressed. Several h5 matrices
  use the one filtered file; otherwise the call raises. A sample title that
  contains spaces can be passed as one string.
- Reusing a server this process did not start checks `/props` and
  `/lora-adapters` again.

- `prepare` drops a stored neighbors record that has no `n_neighbors` before
  it builds the graph. Hao 2021 PBMC carries that record, and Scanpy raised
  `KeyError` on it.
- A marker score keeps every positive gene in the mean. A gene detected in
  fewer than 25% of the cells contributes 0. Dropping that gene let monocyte
  clusters match memory CD4 T cell on `S100a4` and `Cd44` alone.
- A name whose cited markers fail the expression check is not kept when the
  name and ML models agree on another lineage. A marker call that already
  failed is not written again in its place.
- A name under T cell cannot win the marker score unless `CD3D`, `CD3E`,
  `CD3G`, or `TRAC` is detected in at least 25% of the cluster. Mouse symbols
  are `Cd3d`, `Cd3e`, `Cd3g`, and `Trac`. The T cell row itself is unchanged.
- Human memory CD4 T cell uses `SLC4A10` and `TRAV1-2` as negative markers.

## [0.1.1] - 2026-11-09

### Fixed

- A running server is checked with `/props` and `/lora-adapters` when `/proc`
  is unavailable, so Windows and macOS can reuse a server this package did
  not start.
- A `TaskEndpoint` URL calls `/completion` and enables one LoRA adapter for
  that request. When `maxlen` is omitted, name allows 2048 tokens and ML
  allows 8192. An ML prompt does not fit in the name limit.
- A model request waits 600 seconds. `ANNOTUNE_TIMEOUT` or the `timeout`
  argument overrides it. Health checks stay at 2 seconds.
- On Windows, a process check does not call `os.kill`.

### Changed

- The load-model tutorial points Windows users to
  [AnnoTuneLoader](https://github.com/leelieber2025/AnnoTuneLoader).
- A tutorial explains each field of the `run` report. The `Queries` paragraph
  is text to paste into another model.

## [0.1.0] - 2026-11-09

First public release.

### Added

- `read_10x(base, samples)` reads one filtered 10x matrix per sample folder
  (h5 preferred over the mtx directory), makes gene and cell names unique,
  and joins the samples.
- `prepare` fits QC per sample and prints the numeric cutoffs: total counts,
  genes per cell, percent of counts in the top 20 genes, and mitochondrial
  percent together with the 8% floor. Ribosomal and hemoglobin shares are
  recorded and not used as filters.
- `prepare` normalization is `norm="log1p"` (default) or `norm="shifted"`.
  Neighbor-graph batch correction is `none`, `harmony`, `combat`,
  `scanorama`, `scvi`, or `scvi+harmony`.
- Scrublet and SOLO run per sample on the filtered unnormalized counts.
  `doublet_chip` is `"10k"` (0.8% multiplets per 1,000 recovered cells) or
  `"20k"` (0.4% per 1,000). `solo_expected_rates` replaces that rate.
  AmbiDose, when enabled, runs per sample.
- `run` collects three results for each cluster: the marker score, the name
  model, and the ML model. A judgment reads those results and writes the
  title. The title is a dictionary name. The name and ML models run when a
  model chain is passed. The name model copies a name from the marker
  dictionary. The ML model is trained on machine-learning measurements of
  the cluster (differential expression and marker-panel statistics) and
  copies a name from those candidates. A cluster can stay unnamed. A cluster
  with two programs and fewer than 40 cells keeps the name when marker,
  name, and ML agree. An unclear cluster can take the title of the one clear
  cluster it touches.
- `annotate` scores caller-supplied positive and negative markers.
  `evaluate` compares cluster labels with a majority author label.
- The CLI entry point is `annotuneagent`.

### Changed

- GGUF weights are not in the package. They are downloaded separately. The
  tokenizer, marker table, and Cell Ontology snapshot are installed with the
  package.
