# Quickstart

```bash
pip install -e .
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

The call above uses the marker rules only. To also run the name and ML models,
download the weights from the Hub ([lieberlee/annotune-qwen3.5-4b-gguf](https://huggingface.co/lieberlee/annotune-qwen3.5-4b-gguf))
and pass them to `run`. {doc}`tutorials/load_model` covers the
download, the `llama-server` install, and a remote server.

```python
models = ata.local_models("/path/to/annotune-models")
report = ata.run(adata, "leiden", species="human", models=models)
```

`adata.X` must be raw integer counts before `prepare`. Afterwards it is
library-size normalized natural log1p. The default gene list is 2000
`seurat_v3` genes and the Leiden resolution is 0.4. `uns["annotune_prep"]`
records the gene count, the principal-component elbow, and the resolution.

QC thresholds are fit inside each sample (`batch_key`, or `obs["sample"]` when
that column exists). Each sample gets its own cutoffs: log1p total counts, log1p
gene count, and the share of counts in the top 20 genes use 5 median absolute
deviations on both sides (`nmads`); mitochondrial share uses 3 on the upper
side of its logit and must also exceed 8%. Ribosomal and hemoglobin shares are
recorded and not used to filter. `max_pct_mt` is an extra hard cap in percent
(10 means 10%). With `show=True`, the QC step prints the actual cutoffs per
sample, converted back to counts and percent.

`norm="log1p"` (default) scales each cell to 10,000 counts and applies
`log1p`. `norm="shifted"` is the shifted logarithm: each cell is scaled to the
median library size, then `log1p`. Raw counts stay in `layers["counts"]` in
both cases. The CLI flag is `--norm`.

Batch correction is off. `batch_correction="harmony"`, `"combat"`, or
`"scanorama"` changes the neighbor graph used for Leiden and UMAP. Marker
scores still read the log1p matrix. Harmony, Scanorama, scVI, scFair, and
AmbiDose are not installed with the package. If the selected one is missing,
`prepare` prints the install command and leaves the matrix unchanged. ComBat
is the Scanpy implementation.

scVI and doublet removal are independent options. Install `scvi-tools` with
`pip install -e '.[scvi]'`. `batch_correction="scvi"` trains on the raw
`counts` layer (highly variable genes only), stores the latent coordinates in
`obsm["X_scVI"]`, and uses those coordinates for neighbors, Leiden and UMAP.
`adata.X` remains normalized log1p for marker scoring. `n_pcs` sets the latent
dimension in this mode (default 30). Optionally pass `scvi_model_dir` to save
the trained integration model. `batch_correction="scvi+harmony"` runs Harmony
on `obsm["X_scVI"]` and builds neighbors from `obsm["X_scVI_harmony"]`. It
needs at least two batches.

```python
raw_counts = sc.read_h5ad("counts.h5ad")
adata = ata.prepare(
    raw_counts,
    species="human",
    batch_key="sample",
    batch_correction="scvi",
    doublet_method="solo",
    solo_expected_rates={
        "X29_D1": 0.172, "X29_D7": 0.083,
        "Sham_D7": 0.067, "Sham_D1": 0.067,
    },
    scvi_model_dir="saved_scvi_model",
)
```

`doublet_method` is `"none"` (default), `"scrublet"`, or `"solo"`.
Both run separately per sample after QC, on the filtered matrix's unnormalized
UMI counts. They do not read `raw_feature_bc_matrix`; that file is only for
AmbiDose.
`doublet_chip` is `"10k"` (default) or `"20k"`. The expected multiplet rate
follows the Next GEM 3' v3.1 table in 10x CG000422: 0.8% per 1,000 recovered
cells on a 10k chip, and 0.4% per 1,000 on a 20k chip. Both are about 8% at
the rated load. The count is the cells left in that sample after QC.
Scrublet receives the rate as its prior. If the automatic threshold then
calls far fewer doublets than that rate, the cutoff moves to the score whose
overall rate (observed calls divided by the simulated doublet fraction above
the cutoff) is closest. A threshold that already matches is left alone.
SOLO uses the same per-sample rate: its own doublet calls are removed, and so
are the highest-scoring cells up to that fraction. Pass `solo_expected_rates`
only to replace the chip rate, with one fraction for every sample. Omit it
to keep the chip calculation. A universal fraction is unsafe: [10x's standard
and low-throughput v3.1 protocols](https://cdn.10xgenomics.com/image/upload/v1660261285/support-documents/CG000406_TechNote_ChromiumNextGEMSingle_Cell_3-_LT_Kit_Workflow_and_data_overview_RevA.pdf)
have different multiplet rates at the same recovered-cell count, and a 20k
chip is about half the standard-chip rate.
Per-sample `predicted_rate`, `removed_rate`, and `expected_rate` are recorded
in `uns["annotune_prep"]["doublets"]["batches"]`. Surviving cells retain
`solo_score`, `solo_prediction` and `is_doublet`.

The CLI exposes the same choices via `--batch-correction scvi`,
`--doublet-method solo`, `--doublet-chip 10k|20k`, `--scvi-model-dir PATH`,
and `--solo-expected-rates rates.json`. The JSON file maps every sample name to
its rate:

```json
{"X29_D1": 0.172, "X29_D7": 0.083, "Sham_D7": 0.067, "Sham_D1": 0.067}
```

```bash
annotuneagent --input counts.h5ad --output report.json --output-h5ad prepared.h5ad \
  --prepare --species human --batch-key sample --doublet-method solo \
  --solo-expected-rates rates.json --batch-correction scvi --types "T cell"
```

When the CLI writes a prepared `.h5ad`, `uns["annotune_prep"]` is
stored as a JSON string so the nested QC report remains readable by AnnData;
use `json.loads(adata.uns["annotune_prep"])` after reloading it.

```python
print(adata.uns["annotune_prep"])
print(adata.obs["annotune_label"].value_counts(dropna=False))
print(report["dropped_hypotheses"])
print(report["review"])
print(report["marker_dict"])
print(report["marker_validation"])
print(report["annotation_code"])
```

`markers.pdf` groups columns by cell type in the same order as its rows, so
specific markers form diagonal blocks. Candidate genes come from positive
markers in built-in rows that cite a PMID. A gene is plotted only if at least
25% of that type's cells detect it, its log2 fold change is at least 0.5
against all other cells and against the strongest other written type, and its
Bonferroni-adjusted Mann–Whitney p-value is at most 0.05. The report records every
candidate's measurements and the reason it was excluded. A written type with
no passing gene is still drawn, with its best-measured cited genes; its
`marker_status` stays `no_marker_passed_expression_check` and `marker_validation`
keeps `selected: False` for those genes. If the dictionary has no cited
measured gene for any type, no dotplot is written. User-supplied hypotheses without a cited dictionary
row are not labeled as literature-backed markers. The PMID evidence belongs
to the dictionary panel; it is not a separate citation for every gene.

`dropped_hypotheses` names each dictionary type left out and the positive
markers absent from the matrix. `review` lists clusters that still need a
look. Its order is not an accuracy score. A Cell Ontology id is the exact map
row for the written label. An unreviewed id is marked pending.

`annotation_code` contains an editable cluster mapping, `fine_order`, a
`marker_dict`, and UMAP and dotplot calls. Edit the order or labels there and
rerun the snippet to regenerate the plots.

Same written names share one id in `obs["annotune_cluster"]`. The original
Leiden column is kept. A cluster with no written name stays on its own.

Ambient RNA:

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
        "Scramble": "Scramble/raw_feature_bc_matrix.h5",
        "shFGF1": "shFGF1/raw_feature_bc_matrix.h5",
    },
)
```

Without `raw`, that call stops and asks for the unfiltered matrix. A parent folder is split by sample before AmbiDose runs. An AnnData `raw` is split the same way when it has the `batch_key` column.

## How a cluster gets its name

Every cluster with a supported name gets that name. Doubt is recorded in
`annotune_hint` and in `final_reason` on the record; the name is still written.

- `no_marker_passed`: none of the name's cited markers passed the expression
  check. The name stays, the hint says `no marker passed`, and the dotplot
  shows the best-measured cited genes. If the marker call is a different name
  that did pass, that name is written instead.
- `mixed_programs`: a cluster under 40 cells shows two programs. It is named
  only when marker, name and ML all give the same name; the hint lists the
  programs. Otherwise it is left unnamed.
- `near_clear_cluster`: a cluster with an unclear name takes the title of one
  clear cluster it touches on the neighbor graph or the UMAP. A title is clear
  when name and ML wrote the same name, or when no model ran and the marker
  rule wrote it without a hint. A name the model already wrote on the same
  lineage stays, and the hint says which cluster it is next to. A cluster that
  touches two different lineages is not renamed.

`report["queue"]` lists only clusters with no written name. To find named
clusters that still carry a question, filter on `annotune_hint`.
