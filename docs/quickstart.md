# Quickstart

```bash
pip install annotuneagent
```

```python
import scanpy as sc
import annotuneagent as ata

raw_counts = sc.read_h5ad("counts.h5ad")
adata = ata.prepare(raw_counts, species="human", batch_key="sample")
report = ata.run(
    adata, "leiden", species="human", subcluster=False,
    dotplot_path="markers.pdf",
)
```

To start from 10x output, `ata.read_10x(base, samples)` reads one filtered
matrix per sample and joins them. `base` holds one folder per sample name. In
each folder it uses `filtered_feature_bc_matrix.h5`, or the
`filtered_feature_bc_matrix` directory when there is no h5. Barcodes get the
sample name as a suffix, gene and cell names are made unique, and the sample
name goes to `obs["sample"]` (`sample_key` changes the column).

```python
adata = ata.read_10x("outs", ["sample_a", "sample_b"])
adata = ata.prepare(adata, species="mouse", batch_key="sample")
```

To start from a GEO series, `ata.geo_info` reads the series record and lists
each sample title, accession, and characteristics. It does not download
counts. Print that result, choose the samples, and pass the names you want
to `ata.read_geo`. `obs["sample"]` stays the GEO title. `obs["group"]` is the
name you pass. Each characteristic in the GEO record becomes an obs column.
A time point is also stored in `obs["day"]`, and a treatment or stimulus in
`obs["treatment"]`. `obs` adds or replaces those columns when the record does
not name them. Use `sample` or `group` as `batch_key`.

```python
info = ata.geo_info("GSE341410")
adata = ata.read_geo(
    info,
    samples=["B1", "B3"],
    groups="B1: control B3: Day7",
    obs={"day": "B1: 0 B3: 7", "treatment": "B1: none B3: C&AIA"},
)
adata = ata.prepare(adata, species="mouse", batch_key="sample")
```

{doc}`tutorials/gse266598_annotation` runs this path on
[GSE266598](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE266598):
read the series page, download the two calvarial samples, then `prepare` and
`run`.

## Run the name model

The call above collects the marker score only. To also run the name model,
download the weights from the Hub
([lieberlee/annotune-qwen3.5-4b-gguf](https://huggingface.co/lieberlee/annotune-qwen3.5-4b-gguf))
and pass them to `run`. {doc}`tutorials/load_model` covers the download, the
`llama-server` install, and remote servers. If that server was started with
`--api-key`, set `ANNOTUNE_API_KEY` to the same value. It is sent as a bearer
token.

```python
model = "/path/to/Qwen3.5-4B-annotune-v9k-name.Q6_K.gguf"
models = ata.ModelChain(
    name=ata.TaskEndpoint(base=model, maxlen=2048, server="http://127.0.0.1:8080"),
    ml=ata.TaskEndpoint(),
)
report = ata.run(adata, "leiden", species="human", models=models)
```

## Prepare


`adata.X` must hold raw integer counts before `prepare`. Afterward it holds
library-size-normalized natural log1p values. By default `prepare` selects 2000
`seurat_v3` genes and clusters at resolution 0.4. `uns["annotune_prep"]`
records the gene count, the principal-component elbow, and the resolution.

### QC

Thresholds are fit within each sample (`batch_key`, or `obs["sample"]` when
that column exists). For log1p total counts, log1p gene count, and the share of
counts in the top 20 genes, the cutoff is 5 median absolute deviations on both
sides (`nmads`). For mitochondrial share it is 3 on the upper side of
log1p(percent), and the share must also exceed 8%. Ribosomal and hemoglobin
shares are recorded and not used to filter. `max_pct_mt` adds a hard cap in percent (10
means 10%). With `show=True`, `prepare` prints each sample's cutoffs in counts
and percent.

### Normalization

`norm="log1p"` (default) scales each cell to 10,000 counts, then applies
`log1p`. `norm="shifted"` scales each cell to the median library size, then
applies `log1p`. Raw counts stay in `layers["counts"]` in both cases. The CLI
flag is `--norm`.

### Batch correction

Batch correction is off by default. `batch_correction="harmony"`, `"combat"`,
or `"scanorama"` changes the neighbor graph used for Leiden and UMAP. Marker
scores still read the log1p matrix.

`batch_correction="scvi"` needs `pip install 'annotuneagent[scvi]'`. It trains on the
raw `counts` layer (highly variable genes only), stores the latent coordinates
in `obsm["X_scVI"]`, and builds neighbors, Leiden, and UMAP from them.
`adata.X` stays log1p for marker scoring. `n_pcs` sets the latent dimension
(default 30). `scvi_model_dir` saves the trained model.
`batch_correction="scvi+harmony"` runs Harmony on `obsm["X_scVI"]` and builds
neighbors from `obsm["X_scVI_harmony"]`. It needs at least two batches.

### Doublets

```python
adata = ata.prepare(
    raw_counts,
    species="human",
    batch_key="sample",
    batch_correction="scvi",
    doublet_method="solo",
    solo_expected_rates={
        "sample_a": 0.08,
        "sample_b": 0.06,
    },
    scvi_model_dir="saved_scvi_model",
)
```

`doublet_method` is `"none"` (default), `"scrublet"`, or `"solo"`. Both run
per sample after QC, on the filtered matrix's unnormalized UMI counts. They do
not read `raw_feature_bc_matrix`; only AmbiDose does.

`doublet_chip` is `"10k"` (default) or `"20k"`. The expected multiplet rate
follows the Next GEM 3' v3.1 table in 10x CG000422: 0.8% per 1,000 recovered
cells on a 10k chip, and 0.4% per 1,000 on a 20k chip. Both are about 8% at
the rated load. The count is the cells left in that sample after QC.
Scrublet receives the rate as its prior. If the automatic threshold then
calls far fewer doublets than that rate, the cutoff moves to the score whose
overall rate (observed calls divided by the simulated doublet fraction above
the cutoff) is closest. A threshold that already matches is left alone.
SOLO uses the same per-sample rate: it removes its own doublet calls and the
highest-scoring cells up to that fraction. Pass `solo_expected_rates`
to replace the chip rate; it needs one fraction for every sample. Do not use
one rate for all samples: [10x's standard and low-throughput v3.1 protocols](https://cdn.10xgenomics.com/image/upload/v1660261285/support-documents/CG000406_TechNote_ChromiumNextGEMSingle_Cell_3-_LT_Kit_Workflow_and_data_overview_RevA.pdf)
have different multiplet rates at the same recovered-cell count, and a 20k
chip is about half the standard-chip rate.

Per-sample `predicted_rate`, `removed_rate`, and `expected_rate` are recorded
in `uns["annotune_prep"]["doublets"]["batches"]`. Surviving cells retain
`solo_score`, `solo_prediction` and `is_doublet`.

### Command line

The CLI exposes the same choices through `--batch-correction scvi`,
`--doublet-method solo`, `--doublet-chip 10k|20k`, `--scvi-model-dir PATH`,
and `--solo-expected-rates rates.json`. The JSON file maps every sample name to
its rate:

```json
{"sample_a": 0.08, "sample_b": 0.06}
```

```bash
annotuneagent --input counts.h5ad --output report.json --output-h5ad prepared.h5ad \
  --prepare --species human --batch-key sample --doublet-method solo \
  --solo-expected-rates rates.json --batch-correction scvi --types "T cell"
```

When the CLI writes a prepared `.h5ad`, it stores `uns["annotune_prep"]` as a
JSON string so AnnData can write the nested QC report. After reloading, use
`json.loads(adata.uns["annotune_prep"])`.

## Outputs

```python
print(adata.uns["annotune_prep"])
print(adata.obs["annotune_label"].value_counts(dropna=False))
print(report["dropped_hypotheses"])
print(report["review"])
print(report["marker_dict"])
print(report["marker_validation"])
print(report["annotation_code"])
```

### Dotplot

`markers.pdf` orders columns by cell type to match the rows, so specific
markers form diagonal blocks. Candidate genes are the positive markers in
built-in rows that cite a PMID. A gene is plotted only if all of these hold:

- at least 25% of that type's cells detect it;
- its log2 fold change is at least 0.5 against all other cells and against the
  strongest other written type;
- its Bonferroni-adjusted Mann–Whitney p-value is at most 0.05.

The report records every candidate's measurements and the reason it was
excluded. A written type with no passing gene is still drawn, with its
best-measured cited genes. Its `marker_status` stays
`no_marker_passed_expression_check`, and `marker_validation` keeps
`selected: False` for those genes. If no type has a cited, measured gene, no
dotplot is written. Hypotheses you supply without a cited dictionary row are
not labeled as literature-backed. The PMID evidence belongs to the dictionary
panel, not to each gene separately.

### Report fields

- `dropped_hypotheses`: each dictionary type left out because none of its
  positive markers are in the matrix. One missing positive gene stays in the
  panel and scores 0, and the type stays. A missing support gene is omitted.
  A missing negative gene stays on the panel and is not scored.
- `review`: clusters that still need a look. The order is not an accuracy
  score.
- `annotation_code`: an editable cluster mapping, `fine_order`, a `marker_dict`,
  and UMAP and dotplot calls. `drop_clusters` removes one or more clusters
  before those plots. Edit the order, the labels, or that list and rerun the
  snippet to regenerate the plots.

{doc}`tutorials/report` explains every report field. The printed `Queries`
block is text to paste into another model. The heading is the local note.
The paragraph under it is the question. A split adds one block per child,
with the marker judgment and the genes enriched against the other cells.

A Cell Ontology id is the exact map row for the written label. An unreviewed id
is marked pending.

Clusters with the same written title are listed together in `report["merged"]`.
The original Leiden column is kept. A cluster with no title stays on its own.

## Ambient RNA

```python
# One sample: the unfiltered h5, or a raw_feature_bc_matrix directory.
adata = ata.prepare(adata, species="human", denoise=True, raw="raw_feature_bc_matrix.h5")

# Several samples. raw is the parent of one folder per sample name, and each
# folder contains raw_feature_bc_matrix.h5 (preferred) or raw_feature_bc_matrix.
# The sample names are the values in obs[batch_key].
adata = ata.prepare(
    adata,
    species="mouse",
    batch_key="sample",
    denoise=True,
    raw=base,
)
# Or pass each matrix yourself. Keys must match obs["sample"].
adata = ata.prepare(
    adata,
    species="mouse",
    batch_key="sample",
    denoise=True,
    raw={
        "sample_a": "sample_a/raw_feature_bc_matrix.h5",
        "sample_b": "sample_b/raw_feature_bc_matrix.h5",
    },
)
```

Without `raw`, the call stops and asks for the unfiltered matrix. A parent
folder is split by sample before AmbiDose runs. An AnnData `raw` is split the
same way when it has the `batch_key` column.

## How a cluster gets its name

A judgment reads the marker score and the name-model result. The check rule
keeps that name, replaces it with the one supported candidate, or drops it.
The marker score can still name the cluster when the check drops the
proposal. The record's `hint` and `final_reason` say why the title was
written or withheld. Those fields stay on the report.
`check_rule` means the check wrote or withheld the title.

- `cytotoxic_mixture`: the NK panel and the T panel are both specific. The
  cluster stays unnamed so it can be split. A top-ranked `mature NK T cell`
  whose own panel is specific is written as that type. Labels from the split
  are written only when every child cluster has a label.

- `no_marker_passed`: none of the name's cited markers passed the expression
  check. The name stays, the hint reads `no marker passed`, and the dotplot
  shows the best-measured cited genes. When the check rule kept that name, a
  different marker call does not replace it, unless the kept name is
  fibroblast, smooth muscle cell, macrophage, or monocyte and the marker is
  a program-specific type on another lineage. Otherwise a marker call that
  passed is written, and an unchecked marker call is tried once.
- A name the check rule accepted stays when no dictionary panel is detected
  in the majority of the cluster. The same is true when the name model and
  the marker score already agree, unless an ML call is on another lineage.
  The hint lists the enriched genes that are not in those panels. A
  check-accepted fibroblast, smooth muscle cell, macrophage, or monocyte
  yields to a program-specific descendant. Any check-accepted name yields
  to a program-specific marker on another lineage when its own program is off.
- `mixed_programs`: a cluster under 40 cells shows two programs. It stays
  unnamed. The hint lists the programs.
- `near_clear_cluster`: a cluster with an unclear name takes the title of the
  one clear cluster it touches on the neighbor graph or the UMAP. A title is
  clear when the check rule wrote it, or, when no model ran, when the marker
  rule wrote it without a hint. A name the model already wrote on the same
  lineage stays, and the hint names the neighboring cluster. A cluster that
  touches two lineages is not renamed.
- `low_quality`: a cluster is removed only when it has no title of its own
  and at least half of its 20 leading enriched genes are ribosomal
  proteins detected in at least 70% of the other cells. The hint is
  `ribosomal genes only`. A missing title does not remove a cluster. A high
  ribosomal count share does not remove a cluster that already has a title.

`report["queue"]` lists clusters that still have at least one cell without a
title. A cluster whose cells are all named, including by a subcluster split,
is not in `queue`. A titled cluster that still carries a question is in
`report["review"]`.
