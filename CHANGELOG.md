# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

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
