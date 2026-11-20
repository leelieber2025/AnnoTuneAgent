# AnnoTuneAgent Documentation

[![PyPI version](https://img.shields.io/pypi/v/annotuneagent.svg)](https://pypi.org/project/annotuneagent/)
[![PyPI downloads](https://img.shields.io/pepy/dt/annotuneagent.svg?label=PyPI%20downloads)](https://pepy.tech/project/annotuneagent)
[![Python versions](https://img.shields.io/pypi/pyversions/annotuneagent.svg)](https://pypi.org/project/annotuneagent/)
[![CI](https://github.com/leelieber2025/AnnoTuneAgent/actions/workflows/tests.yml/badge.svg)](https://github.com/leelieber2025/AnnoTuneAgent/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/leelieber2025/AnnoTuneAgent/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24092828.svg)](https://doi.org/10.5281/zenodo.24092828)

AnnoTuneAgent annotates clustered scRNA-seq. `prepare` runs QC, gene
selection, and Leiden clustering. `run` collects a marker score and a
name-model result. The check rule then keeps that name, replaces it, or drops
it. The marker score can still name the cluster when the check drops the
proposal. A cluster can stay unnamed. Clusters with the same title are merged.

The name model is one merged Q6_K GGUF, `Qwen3.5-4B-annotune-v9k-name.Q6_K.gguf`.
It runs locally and copies
a name from the marker dictionary hits (`dictionary_hits`). The check is not
a model. A positive marker supports a type when its log fold change is above
0.5 and it is detected in at least 25% of the cluster. A negative marker
blocks a type only when it is that common and its log fold change is at least
1. The proposal stays when at least two markers support it and none block it.
It is replaced only when it is not supported and exactly one other candidate
is.

The name model runs when `models` is passed. Without it, `run` uses the marker
score only. The weights are on the Hugging Face Hub:
[lieberlee/annotune-qwen3.5-4b-gguf](https://huggingface.co/lieberlee/annotune-qwen3.5-4b-gguf).

```python
import annotuneagent as ata

adata = ata.prepare(adata, species="human")
report = ata.run(adata, "leiden", species="human", subcluster=False)
```

`prepare` expects raw integer counts. AmbiDose ambient subtraction is off
unless you pass `denoise=True` and the unfiltered droplet matrix.

| To | Pass |
|----|------|
| Name the sample column | `batch_key="sample"` |
| Remove doublets | `doublet_method="scrublet"` or `"solo"` |
| Set the chip type | `doublet_chip="10k"` (default) or `"20k"` |
| Set SOLO removal rates per sample | `solo_expected_rates={"sample_a": 0.08, "sample_b": 0.06}` |
| Cluster on scVI latent space | `batch_correction="scvi"` |
| Run Harmony on the scVI latent space | `batch_correction="scvi+harmony"` |
| Subtract ambient RNA | `denoise=True` and per-sample `raw`: a parent folder, or a dict keyed by sample |
| Change the Leiden resolution | `resolution=0.8` |
| Change the highly variable gene count | `n_top_genes=3000` |
| Use scFair for genes and resolution | `scfair=True` |

By default `prepare` selects 2000 genes with Scanpy `seurat_v3` and clusters at
resolution 0.4. A resolution you pass is used as given. With `scfair=True`,
scFair supplies the gene list and the resolution is half of its suggestion.

| Page | Contents |
|------|----------|
| {doc}`installation`, {doc}`quickstart` | Install, then a first call |
| {doc}`tutorials/pbmc8k_annotation` | PBMC 8k: `prepare`, then the name model and the check rule |
| {doc}`tutorials/gse266598_annotation` | GEO GSE266598: download, then `prepare` and annotation |
| {doc}`tutorials/load_model` | Download the weights and run a local or remote model server |
| {doc}`tutorials/report` | What each part of the `run` report is for, including the text to paste |
| {doc}`api/index` | Function signatures |
| {doc}`marker_sources` | Marker reference sources and verification decisions |

```{toctree}
:maxdepth: 1
:hidden:

installation
quickstart
tutorials/index
api/index
marker_sources
```

## Author

**Zhao Li (李钊)** <leelieber@gmail.com>
