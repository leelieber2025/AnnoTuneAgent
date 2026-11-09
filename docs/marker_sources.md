# Marker dictionary sources and verification

The built-in dictionary is [`markers.tsv`](../src/annotuneagent/markers.tsv).
Source databases help find candidate cell types and genes. A database hit alone
is not evidence that a gene is a positive RNA marker in the specified species,
tissue, and cell type. Each accepted row records an original-study PMID in its
`evidence` field. The final dotplot should additionally check expression in
the user's own data; literature evidence cannot guarantee detectability in
every assay or sample.

## Downloaded candidate sources

Downloaded and checked on 2026-11-08. SHA-256 identifies the exact local
snapshot examined. Raw source files are not distributed with this project.

| Source | Download and checked snapshot | Fields used | How it is used |
|---|---|---|---|
| [Cell Taxonomy](https://ngdc.cncb.ac.cn/celltaxonomy/download), [database paper](https://pubmed.ncbi.nlm.nih.gov/36161321/) | [`Cell_Taxonomy_resource.txt`](https://download.cncb.ac.cn/celltaxonomy/Cell_Taxonomy_resource.txt), 117,817,373 bytes, SHA-256 `1ede722412dc0008fbf2f4fd16b320f5bad8acaa7980b872282b4cbfa1974a81` | `Species`, `Tissue_standard`, `Specific_Cell_Ontology_ID`, `Cell_Marker`, `PMID`, `Source` | 226,222 records checked. `Source` distinguishes original Cell Taxonomy curation from imported CellMarker, PanglaoDB, OnClass, etc. Imported rows are leads, not independent confirmation. Follow the cited study and check its organism, assay, and cell population. See [database help](https://ngdc.cncb.ac.cn/celltaxonomy/help) for source definitions and use terms. |
| [Human Reference Atlas ASCT+B](https://humanatlas.io/asctb-tables) | [All tables v2.5 JSON](https://cdn.humanatlas.io/hra-asctb-json-releases/hra-asctb-all.v2.5.json), SHA-256 `5644a7c93d9e6418d07240ea1b5671eed43f2a0c625f959c2880b1f8f8b1ea01` | organ, `cell_types`, `biomarkers_gene`, `biomarkers_protein`, `references`, ontology IDs | Human organ and CL coverage check. Gene and protein biomarkers are kept distinct. An expert table may cite an organ-level source rather than direct RNA evidence for each gene. |
| [scCATCH CellMatch](https://github.com/ZJUFanLab/scCATCH), [method paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7031312/) | [`cellmatch.rda`](https://raw.githubusercontent.com/ZJUFanLab/scCATCH/master/data/cellmatch.rda), 49,560 rows, SHA-256 `117ce4a1b1cdffb180f2d2cf1c05b66dbbf14d7039ea7488717ea771d96c5518` | `species`, `tissue`, `celltype`, `gene`, `resource`, `pmid` | Candidate discovery. CellMatch integrates other marker sources; a matching row and its upstream source count as one evidence chain. The URL tracks `master`; use the checksum to identify the checked file. |

## Additional sources inspected

| Source | Role and limit |
|---|---|
| [CellMarker 3.0](https://doi.org/10.1093/nar/gkag1040) | Candidate database with curated and computationally inferred records; inspect each record's provenance before inclusion. |
| [PIASOmarkerDB](https://piaso.org/marker/faq) | Study-context specificity scores. Useful for ranking candidates, not a substitute for original-study evidence or expression in the user's matrix. |
| [Human Protein Atlas downloads](https://www.proteinatlas.org/about/download) | Human single-cell expression check. Tissue and cell-type definitions may differ from this project's CL labels. |
| [CellKb](https://www.cellkb.com/knowledgebase) | Literature signature discovery. Check the linked paper and access terms before importing a signature. |
| [Cell Ontology](https://github.com/obophenotype/cell-ontology) | CL ID and label check. The repository's pinned release and checksum are recorded in `src/annotuneagent/ontology/ontology_map.jsonl`; ontology membership does not establish marker expression. |

## Decisions from this review

| Cell type | Decision | Direct evidence and caveat |
|---|---|---|
| Intestinal interstitial cell of Cajal, human and mouse | Added `KIT;ANO1` / `Kit;Ano1` | [PMID:19372102](https://pubmed.ncbi.nlm.nih.gov/19372102/) colocalized both proteins across human and mouse GI regions and showed mast cells are KIT-positive but ANO1-negative. Human colon nuclei also express both transcripts ([PMID:37206377](https://pubmed.ncbi.nlm.nih.gov/37206377/)). In intestine use both genes and exclude mast-cell/leukocyte clusters; an epithelial ANO1 signal alone is insufficient. |
| Kidney parietal epithelial cell, human | Added `CLDN1;CFH;ALDH1A2` | The [adult human kidney single-cell annotation](https://www.celltype.info/project/588/dataset/1327/label/276142) lists all three as PEC assignment markers; [PMID:39360029](https://pubmed.ncbi.nlm.nih.gov/39360029/) checks ALDH1A2 enrichment across human kidney datasets. ALDH1A2 is detected in only about 35% of PECs in one scRNA dataset, so absence in one cell is weak negative evidence. |
| Kidney parietal epithelial cell, mouse | Added `Cldn1;Pax8` | A [primary mouse glomerular single-cell study, PMID:39177649](https://pubmed.ncbi.nlm.nih.gov/39177649/), identifies a Pax8-positive, Cldn1-positive PEC cluster. Pax8 is also expressed in renal tubules; interpret the combination within glomerular context. |
| Merkel cell | Held for a later review | [PMID:25468937](https://pubmed.ncbi.nlm.nih.gov/25468937/) supports mouse Krt20 and developmental markers, while human studies commonly use KRT20 protein. Rare normal adult skin cells are often absent from generic scRNA atlases. Need a direct adult human RNA comparison before adding a general dictionary row. |
| Sinoatrial pacemaker cell | Held for a later review | HRA lists HCN4/GJC1, but HCN4 overlaps other conduction cells and many supporting single-cell datasets are developmental. Need a matched adult, species-specific RNA panel before a general row. |

One rejected database example: Cell Taxonomy's imported `OnClass` records for
parietal epithelial cells include `Emcn`, `Esam`, `Egfl7`, and `Ptprb`. These are
vascular genes and were not accepted as positive PEC markers. This illustrates
why ontology matching and a database PMID are insufficient without checking
the source study and its marker-to-cell assignment.
