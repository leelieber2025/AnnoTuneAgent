# AnnoTuneAgent

[![PyPI version](https://img.shields.io/pypi/v/annotuneagent.svg)](https://pypi.org/project/annotuneagent/)
[![PyPI downloads](https://img.shields.io/pepy/dt/annotuneagent.svg?label=PyPI%20downloads)](https://pepy.tech/project/annotuneagent)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/annotuneagent.svg)](https://anaconda.org/bioconda/annotuneagent)
[![Bioconda downloads](https://img.shields.io/conda/dn/bioconda/annotuneagent.svg?label=Bioconda%20downloads)](https://anaconda.org/bioconda/annotuneagent)
[![Total downloads](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/leelieber2025/AnnoTuneAgent/badges/downloads.json)](https://github.com/leelieber2025/AnnoTuneAgent/blob/badges/history.json)
[![Python versions](https://img.shields.io/pypi/pyversions/annotuneagent.svg)](https://pypi.org/project/annotuneagent/)
[![Documentation Status](https://readthedocs.org/projects/annotuneagent/badge/?version=latest)](https://annotuneagent.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/leelieber2025/AnnoTuneAgent/actions/workflows/tests.yml/badge.svg)](https://github.com/leelieber2025/AnnoTuneAgent/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24092828.svg)](https://doi.org/10.5281/zenodo.24092828)

Documentation: [Read the Docs](https://annotuneagent.readthedocs.io/en/latest/).

**Cluster-level cell-type annotation for single-cell RNA-seq. One fine-tuned
language model runs locally, and a check rule reads the marker measurements.**

Each cluster gets a marker score and a name-model result. The check rule
keeps that name, replaces it, or drops it. The marker score can still name
the cluster when the check drops the proposal. The report keeps the name,
the check result, and the reason.

## What is fine-tuned

The current name model is one Q6_K GGUF,
`Qwen3.5-4B-annotune-v9k-name.Q6_K.gguf`. It is loaded with `-m` and no
LoRA. If that server was started with `--api-key`, set `ANNOTUNE_API_KEY`.
The same value is sent as a bearer token.

| Step | What it does | Input | Output |
| --- | --- | --- | --- |
| **name** | Choose a cell-type name from the marker dictionary | Cluster marker evidence and `dictionary_hits` | A name copied from `dictionary_hits` |
| **check** | A fixed rule, not a model | The proposed name and each candidate's marker measurements | Keep the name, replace it with the one supported candidate, or drop it |

The name model is trained to **copy** a name from `dictionary_hits`, not to
free-write one. A positive marker supports a type when its log fold change is
above 0.5 and it is detected in at least 25% of the cluster. A negative
marker blocks a type only when it is that common and its log fold change is
at least 1. The proposal stays when at least two markers support it and none
block it. It is replaced only when it is not supported and exactly one other
candidate is. Otherwise it is dropped, and the marker score can still name
the cluster. A name with no marker that passes expression and specificity
validation is flagged (`no_marker_passed`) instead of being trusted silently.

The weights are on the Hugging Face Hub:
[lieberlee/annotune-qwen3.5-4b-gguf](https://huggingface.co/lieberlee/annotune-qwen3.5-4b-gguf)
(preview). Inference is local: no cluster data leaves the machine.

## Pipeline

```text
raw counts ─ prepare ─▶ QC, doublets, normalization, batch correction, Leiden
           ─ run ─────▶ per cluster:
                          1. marker score        (built-in human / mouse dictionary)
                          2. name model          (one GGUF, copies from dictionary_hits)
                          3. check rule          (keeps, replaces, or drops that name)
                          ▼
                        title + status + reason + hint
                          ▼
                        validated marker panels, dotplot, report, annotated h5ad
```

- The check reads the measurements. Mixed programs and names that fail the
  marker check stay flagged (`mixed_programs`, `near_clear_cluster`,
  `subcluster_labels`, `no_marker_passed`, `check_rule`).
- Panels come from cited dictionary markers. A gene stays on the plot when
  it is enriched in that label and specific against the other labels.
- The report stores the name-model answer, the check result, the panel, and
  the reason.

The name model runs when `models` is passed, or when the command line sets
`--base`, `--name-url`, or `--name-adapter`. Without that, `run` uses the
marker score alone.

## Install

Python 3.10–3.13.

```bash
pip install annotuneagent
# or: conda install -c conda-forge -c bioconda annotuneagent
```

Local GGUF models need `pip install 'annotuneagent[models]'`, which installs
the tokenizer stack (`transformers` and `jinja2`). Optional methods:
AmbiDose (`denoise=True`, [install](https://github.com/leelieber2025/AmbiDose)),
scVI and SOLO (`annotuneagent[scvi]`), scFair (`annotuneagent[scfair]`),
Harmony, and Scanorama. If you select one that is missing, `prepare` prints
the install command and leaves the matrix unchanged. See
[Installation](docs/installation.md).

From a checkout:

```bash
git clone https://github.com/leelieber2025/AnnoTuneAgent.git
cd AnnoTuneAgent
pip install -e ".[dev]"
```

## First run

`prepare` expects raw integer counts. `run` expects finite, nonnegative,
library-size-normalized natural log1p expression with Scanpy's
`adata.uns["log1p"]`, which `prepare` writes.

```python
import annotuneagent as ata

adata = ata.read_10x("outs", ["sample_a", "sample_b"])
adata = ata.prepare(adata, species="human", batch_key="sample")
report = ata.run(adata, "leiden", species="human", subcluster=False)
```

`read_10x` takes one filtered matrix per sample folder (`*.h5`, or the mtx
directory when there is no h5). Scrublet and SOLO run per sample on those
filtered counts. `doublet_chip="10k"` (default) uses 0.8% multiplets per
1,000 recovered cells, and `"20k"` uses 0.4%.

`geo_info("GSE341410")` lists the samples in a GEO series. `read_geo` then
downloads the ones you name. `groups` sets `obs["group"]`. Characteristics
such as day and treatment are stored on `obs`, and `obs={...}` adds columns
the record does not name.
[docs/tutorials/gse266598_annotation.ipynb](docs/tutorials/gse266598_annotation.ipynb)
follows one series from the GEO page through annotation.

To score a caller-supplied list instead of the built-in dictionary:

```python
records = ata.annotate(
    adata,
    cluster_key="leiden",
    hypotheses=[
        ata.Hypothesis(
            "CD4-positive, alpha-beta T cell",
            positive=("CD3E", "CD4"),
            negative=("CD8A",),
        ),
        ata.Hypothesis(
            "CD8-positive, alpha-beta T cell",
            positive=("CD3E", "CD8A"),
            negative=("CD4",),
        ),
    ],
)
```

A label needs positive marker enrichment. A negative marker counts against a
hypothesis only when the cluster is enriched for it relative to the other
cells. Scores are operational, not probabilities.

The built-in table is `markers.tsv`: human and mouse, each with a
tissue-agnostic core and organ pages. Tissue rows stay hidden until `tissue`
matches. Symbols are not translated across species. `evaluate` compares
records with a per-cell author label: the reference for a cluster is the
majority label, a tied majority is left unmatched, and a parent name counts
as a coarse match.

```bash
annotuneagent \
  --input clustered.h5ad \
  --cluster-key leiden \
  --species human \
  --output result.json
```

`--base` with the merged GGUF runs the name model on one local server and
requires `--output-h5ad`. The check rule then reads the marker measurements.
The weights are on the Hugging
Face Hub: [lieberlee/annotune-qwen3.5-4b-gguf](https://huggingface.co/lieberlee/annotune-qwen3.5-4b-gguf).
The tokenizer is installed with the package. See
[docs/tutorials/load_model.ipynb](docs/tutorials/load_model.ipynb).

## Status

**0.1.7 (Alpha).** Import as `import annotuneagent as ata`.

## Citation

For the software record, cite the [Zenodo DOI](https://doi.org/10.5281/zenodo.24092828).
Pin the package version used in the analysis, for example `annotuneagent==0.1.7`.

## License

Software: [Apache License 2.0](LICENSE).

## Author

**Zhao Li (李钊)**
Email: [leelieber@gmail.com](mailto:leelieber@gmail.com)
