# Read the report

`run` returns a dictionary and prints a `Queries` block. Each block is one
cluster that still has a question: it has no title, or the title is written
and `hint` is set. AnnoTuneAgent does not send this text to the name model,
the ML model, or any other service. Copy the paragraph under the heading and
paste it into another model.

## The heading

```text
cluster 0  monocyte  no marker passed
```

The three fields are the cluster id, the written title, and the hint. A
cluster with no title prints `no label` in the middle. The heading stays in
the notebook. It is not part of the question.

`no marker passed` means the title `monocyte` was written, and none of its
cited markers passed the expression check. The name stays. The dotplot still
draws the best-measured cited genes for that type.

```text
cluster 2  CD14-positive monocyte  conflict: marker neutrophil
```

The written title is `CD14-positive monocyte`. The marker score named
`neutrophil`, and that name is not on the same Cell Ontology lineage as the
title. The hint records the other call. The same line can name the name model
or the ML model, for example `conflict: ML pericyte`. Several disagreements
are joined with commas.

## The paragraph to paste

```text
human, 1728 cells. What cell type is this cluster?
Enriched versus other cells: HLA-DRA (in 1.00, out 0.62, logfc 2.46); HLA-DRB1 (in 0.99, out 0.65, logfc 2.15); CD74 (in 1.00, out 0.93, logfc 1.84); LGALS2 (in 0.74, out 0.15, logfc 3.40); HLA-DPB1 (in 0.95, out 0.56, logfc 2.08); HLA-DRB5 (in 0.94, out 0.53, logfc 1.96); PID1 (in 0.70, out 0.19, logfc 2.78); CPVL (in 0.84, out 0.36, logfc 2.07); HLA-DPA1 (in 0.96, out 0.58, logfc 1.98); LYZ (in 1.00, out 0.65, logfc 3.01).
Also detected: CD14 (in 0.87, out 0.42, logfc 1.71); FCN1 (in 0.99, out 0.59, logfc 1.83); CSF1R (in 0.62, out 0.31, logfc 1.16); S100A8 (in 1.00, out 0.57, logfc 2.39); VCAN (in 0.94, out 0.50, logfc 1.59); S100A4 (in 1.00, out 0.95, logfc 0.81); ANXA1 (in 0.81, out 0.69, logfc 0.35).
```

The first line is the species, the tissue when `run` was given one, the
number of cells, and the question. Paste from this line through `Also
detected`.

`Enriched versus other cells` lists up to 10 genes that are higher in this
cluster than in the other cells, in Wilcoxon order. `in` is the fraction of
cells in the cluster where the gene is detected. `out` is that fraction in
the other cells. `logfc` is the Scanpy log fold change.

`Also detected` lists markers of dictionary programs that are on in this
cluster, detected in at least half of its cells, and not already in the
enriched list. The list stops at 16 genes.

A reply from the other model is not written back into `adata` or the report.
Use it when you decide whether to keep or change the title in
`annotation_code`.

The same paragraph is `query` on `report["clusters"]` and on
`report["review"]`.

## One row per cluster

`report["clusters"]` has one row per cluster. The columns stay separate. A
title is not a vote count.

| Column | What it is |
|---|---|
| `cluster` | Leiden id |
| `cells` | Cells in that cluster |
| `marker` | Name from the marker score, or none |
| `name` | First name copied by the name model. `failed` when the call failed. The action when the reply named nothing |
| `ml` | First name copied by the ML model. `escalate` when it abstains. `failed` when the call failed |
| `label` | Title written onto the cells. Empty when the cluster stays unnamed |
| `hint` | The question printed after the title. Empty when there is nothing to ask |
| `query` | The paragraph to paste. Present when the cluster is unnamed or `hint` is set |
| `reason` | Why the title was written or withheld |
| `ontology` | Cell Ontology id for the written title. `pending` means the id is not a reviewed row |

`adata.obs["annotune_label"]` is the title. `adata.obs["annotune_hint"]` is
the hint on every cell of that cluster. `adata.obs["annotune_final_status"]`
is `named`, `review`, or `escalate`. Clusters that share a title also share
one id in `adata.obs["annotune_cluster"]`. The Leiden column is kept.

## What to read first

`report["review"]` is the clusters to read first. The order is a reading
order, not an accuracy score.

| Priority | Which clusters |
|---|---|
| 0 | No title |
| 1 | A conflict, or close marker scores |
| 2 | A model call failed |
| 3 | Fewer than 20 cells |
| 4 | Any other hint |

`report["queue"]` lists only clusters with no title. A titled cluster with a
hint is in `review`, and on `annotune_hint`, and it is not in `queue`.

## The rest of the report

| Field | What it is |
|---|---|
| `records` | The working record for each cluster, including model rounds and the marker ranking |
| `dropped_hypotheses` | Dictionary types left out because none of their positive markers are in the matrix |
| `subclusters` | The second split of a cluster that was not settled. Empty when `subcluster=False` |
| `arbitration` | Clusters that were not given the status `named` |
| `merged` | Titles that were given one `annotune_cluster` id |
| `all_labeled` | True when every cell has a title |
| `written` | True when at least one cell has a title |
| `label_key` | The `obs` column that holds the title. `None` when nothing was written |
| `marker_dict` | Genes drawn in the dotplot, in column order |
| `marker_validation` | Measurement of each cited gene, and why it was kept or left out |
| `marker_status` | Per written type. `no_marker_passed_expression_check` is the check behind `no marker passed` |
| `marker_thresholds` | The cutoffs used for that check |
| `dotplot` | Path of the PDF when `dotplot_path` was set |
| `doublets` | Doublet summary stored by `prepare` |
| `tissue` | Tissue page passed to `run` |
| `tissues` | Tissue pages available for the species |
| `annotation_code` | Python that redraws the UMAP and the dotplot from the written titles |

`annotation_code` copies the titles into `cluster2celltype`. Clusters with no
title are left out of that map and listed under `cluster_choices` when the
record has candidate names. Edit the map or `fine_order`, then run the
snippet on the annotated object. {doc}`../quickstart` records the expression
cutoffs used for the dotplot.
