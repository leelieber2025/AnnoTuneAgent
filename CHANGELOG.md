# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [0.1.8] - 2026-11-20

### Changed

- A check-accepted fibroblast, smooth muscle cell, macrophage, or monocyte
  yields to a program-specific marker on another lineage, and to a
  program-specific descendant. One broad title does not replace another.
  Any check-accepted name yields when its own program is off and the marker
  is program-specific on another lineage.
- A name the check rule accepted stays when no dictionary panel is on.
  The same keep applies when the name model and the marker score agree,
  unless an ML call is on another lineage. The hint lists the uncovered genes.
- Tongue maps to the skin page. An author tissue that matches one page,
  such as large intestine, uses that page's aliases.
- Dictionary panels with one positive gene now have a second, except human
  juxtaglomerular cells (`REN`). Prostate basal cells add KRT14 and KRT15.
  Non-keratinized squamous epithelium is KRT4 and KRT13. Tenocytes add COMP
  and CILP. Lyve1-positive macrophages use LYVE1 and F13A1, with PRG4 as a
  negative. Mouse pericytes include Notch3. Superficial-zone chondrocytes
  use PRG4, SFRP1, and COL2A1. Mammary basal cells use KRT5, KRT14, and
  ACTA2. Smooth muscle treats KRT5 and KRT14 as negatives. Luminal mammary
  epithelium adds KRT19.
- Marker dot plots use square cells. The load-model tutorial downloads the
  one merged GGUF. Docs include the Bioconda install and weekly download
  badges.

## [0.1.7] - 2026-11-19

### Fixed

- Remove the unused merged-id list in `merge_same_labels`. `ruff` failed the
  0.1.6 tests job on that assignment. Annotation behavior matches 0.1.6.

## [0.1.6] - 2026-11-19

### Changed

- `run` calls the name model only. The check rule keeps that name, replaces
  it with the one other supported candidate, or drops it. A supporting marker
  has logfc above 0.5 and pct_in of at least 0.25. A blocking negative has
  pct_in of at least 0.25 and logfc of at least 1. A cluster whose name the
  check rule accepted is not split.
- The name prompt lists the top 35 Wilcoxon genes. An enriched support gene
  is appended only after a positive marker already qualifies. Low or
  undetected genes stay out.
- The name weight is one merged GGUF,
  `Qwen3.5-4B-annotune-v9k-name.Q6_K.gguf`. Pass it as `TaskEndpoint(base=...)`
  or `--base`. The server context for that call is 2048.
- `adata` gains `annotune_label`, and `annotune_sub` only when a cluster
  stays split. Status, hints, and merged cluster ids stay on the report.
- `write_h5ad` is no longer part of the public API. `--output-h5ad` still
  writes the file and stores the QC report as JSON.
- The marker dictionary adds rare types, including plasmablast,
  myofibroblast, BEST4+ enterocyte, mouse ILC1, muscle fiber subtypes,
  bone-marrow stroma, juxtaglomerular cells, urothelial basal and umbrella
  cells, spermatogenic stages, mammary luminal subsets, adrenal zones,
  decidual stroma, mouse trophoblast, and thymic epithelium. Human prostate
  luminal cells use ACP3. Human non-peptidergic nociceptors no longer treat
  CALCA as a negative. Mouse corneal epithelium uses Krt12 and Krt5.
  Colonocyte keeps CL:1000347.

### Fixed

- A cluster removed as `ribosomal genes only` still prints its enriched-gene
  paragraph.
- A later split can write `parent.child` ids after the subcluster column has
  been turned into a categorical.
- Subcluster PCA on a sparse matrix uses `arpack`.

## [0.1.5] - 2026-11-17

### Changed

- Mitochondrial QC uses 3 median absolute deviations on the upper side of
  log1p(percent), fit inside each sample. A cell is removed only when its
  mitochondrial share also exceeds 8%. `max_pct_mt` is still an optional
  hard cap.
- A title that the name model and the ML model already agree on stays written
  when its cited markers fail the expression check. A different marker call
  does not replace it, unless that marker program is specific and the agreed
  program is not on. Otherwise the hint stays `no marker passed`.
- The cluster query lists up to 20 enriched genes. It listed 10.
- A cluster is removed only when it has no title of its own and at least
  half of its 20 leading enriched genes are ribosomal proteins
  detected in at least 70% of the other cells. A missing title alone does
  not remove it. A cluster that already has a title stays. The hint is
  `ribosomal genes only`.
- `prepare` with `show=True` no longer draws the highly variable gene plot
  or the two scatters of total counts against gene count and mitochondrial
  percent.
- `run` no longer draws a UMAP colored by `annotune_cluster`. The merged ids
  stay in `obs["annotune_cluster"]`. The label UMAP is unchanged.
- `annotation_code` includes `drop_clusters`. The list starts empty. Cluster
  ids placed there are removed before the labels and plots. An empty Leiden
  category left by that removal is dropped. A cell type with no remaining
  cells is left out of the UMAP legend and the dotplot.
- A cluster whose cells were removed is not split afterwards.
- A split prints one Queries block for each remaining child, and adds that
  child to `report["clusters"]`. The paragraph gives the marker judgment and
  the genes enriched against the other cells.
- A cluster whose name and ML calls are endothelial cell and pericyte is
  split even when a child stays unlabeled. A labeled child names every
  cell in that child. Children of a lineage split are scored against every
  other cell. When pericyte and smooth muscle cell are both specific, neither
  name is written. The hint is `pericyte or smooth muscle cell`. When only
  one of those programs is specific, that name is written.
- Subclustering a cluster of fewer than 200 cells tries resolutions 0.1
  and 0.2. A cluster of 200 to 499 cells tries 0.2 and 0.4. Larger
  clusters still try 0.6, 1.0, and 1.4. Silhouette chooses. On 141 cells,
  0.2 keeps two groups.
- A split between two names keeps at most two child clusters. Resolutions
  that make more groups are skipped, and 0.05 and 0.1 are tried first.
  Every cell in a labeled child takes that child's name.

## [0.1.4] - 2026-11-16

### Fixed

- A cytotoxic mixture that is not fully split stays `review`, with reason
  `cytotoxic_mixture` and a hint that NK and T programs are both enriched.
  `evaluate` scores a workflow record by `written_label`, so that cluster is
  not counted as a marker hit. Pass `prediction_key="annotune_label"` to score
  the labels written on the cells.
- Child labels of a cytotoxic mixture are written only when every child has a
  label.
- `annotate` skips a support gene that is absent from the matrix.
- Adapter ids from `/lora-adapters` are cached only after the name file and
  the ml file are both parsed. A failed read, or a list that does not name
  both adapters, is tried again on the next request. Until then the load
  order is name 0 and ml 1.

### Changed

- `evidence_agreement` that is not program-specific stays written. The hint
  is `weak agreement on {label}` and the cluster stays in the review list.
  A one-gene panel cannot meet the specific test, so that agreement is left
  unchanged. When no candidate panel is detected in the majority, the status
  is `out_of_dictionary`, no name is written, and the hint lists genes that
  are high and not in the dictionary panels.
- A cluster with no title takes a name that is already written on a cluster
  it touches on the UMAP. Connection is the distance between the closest
  cells, within two typical neighbor spacings. The shorter distance wins.
  A name withheld as `out_of_dictionary` is not copied. The hint records the
  neighbor and the distance.
- A subcluster whose cells all receive one title is dropped. The parent
  keeps that title, and the child ids are not left as unresolved clusters.
  A cytotoxic mixture is left split.
- A cluster is split when the name model and the ML model name different
  lineages, including one the marker score already named. The split is kept
  only when the children receive more than one title.
- When no panel is detected and the calls are endothelial cell and pericyte,
  endothelial cell is written if an endothelial gene is enriched and no
  pericyte program gene is. The hint is `endothelial cell rather than pericyte`.
- Each ranking row records `presence`, the mean fraction of cluster cells
  that express the positive markers. Presence does not choose the name.
- Every cluster gets an enriched-gene query.
- `read_geo` stores each GEO characteristic on `obs`. A time point is also
  `obs["day"]`, and a treatment or stimulus is also `obs["treatment"]`. Those
  words are read from the sample title when the record does not name them.
  `obs={...}` adds or replaces sample-level columns with the same syntax as
  `groups`.
- The docs state that one missing positive gene stays in the panel and scores
  0. `ANNOTUNE_API_KEY` is documented for a server started with `--api-key`.
- A cluster whose top marker panel is a specific `mature NK T cell` is named
  as that type. NK and T programs that are both specific still withhold a
  title when another type ranks first.
- Limb `Schwann cell` uses `MOG`/`Mog` as the oligodendrocyte negative marker.
  `PLP1`/`Plp1` stays off that negative list because Schwann cells express it.

## [0.1.3] - 2026-11-11

### Added

- Dictionary rows for macrophage, mature NK T cell, choroid plexus epithelium,
  kidney connecting tubule epithelium, retinal horizontal and amacrine cells,
  skeletal muscle myoblast, ovarian stroma, Merkel cell, small-intestine
  transit amplifying cell, limb Schwann cell, and brain innate lymphoid cell.
- `evaluate(..., raw_label_key=...)`. A mapped name is the reference only when
  it is the majority of every author-labeled cell. An unmapped raw majority
  stays unmapped.
- `write_h5ad` saves a prepared AnnData with its QC report.
- `ANNOTUNE_API_KEY` is sent as a bearer token.

### Changed

- A marker score caps each gene's log2 fold change at 3. A fired negative marker
  subtracts half of one positive gene's share and still blocks a hard marker
  label.
- When a call touches both CD4 and CD8, the written title is T cell. A cluster
  whose NK and T panels are both specific is a cytotoxic mixture. A tissue
  macrophage or alveolar fibroblast named by a model is kept when its own
  panel is on. Effector-memory CD8 is not written when a MAIT marker is
  enriched.

### Fixed

- `dictionary_hits` order no longer depends on the input file name. The
  command line and the Python API use the same fixed seed.
- A negative marker that is absent from the matrix stays on the ML panel.
  `n_negative_unmeasured` counts it, and the marker score does not.
- `annotuneagent[models]` installs `jinja2`, which the chat template needs.
  `run` with model endpoints checks `transformers` and `jinja2` before
  scoring instead of failing at the first cluster.
- Model runs no longer stop on a one-cell cluster, an integer cluster
  column, or unused categories. Wilcoxon skips clusters with fewer than two
  cells; their model calls are recorded as failed.
- A label needs a positive gene that is enriched and detected in 25% of the
  cluster, the same rule as the score.
- A model URL with a trailing slash no longer posts to `//completion` (404).
- A server not started by this process is called with the adapter ids its
  `/lora-adapters` file names give, so `--lora ml.gguf,name.gguf` is not
  swapped. A server started with repeated `--lora` flags is recognized.
- `TaskEndpoint(system=...)` is used on the URL path.
- The report shows `escalate` for an ML escalation instead of `null`.
- `--hypotheses` reads UTF-8 with or without a BOM, accepts `support` and
  `requires_any`, and rejects non-list gene fields. `Hypothesis("x",
  positive="CD3E")` is one gene, not four letters.
- `read_geo` keeps an uncompressed `features.tsv` or `genes.tsv`.
- `evaluate` treats a cluster with no record as uncertain.

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
