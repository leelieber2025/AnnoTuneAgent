# AnnoTuneAgent Documentation

[![PyPI version](https://img.shields.io/pypi/v/annotuneagent.svg)](https://pypi.org/project/annotuneagent/)
[![PyPI downloads](https://img.shields.io/pepy/dt/annotuneagent.svg?label=PyPI%20downloads)](https://pepy.tech/project/annotuneagent)
[![Python versions](https://img.shields.io/pypi/pyversions/annotuneagent.svg)](https://pypi.org/project/annotuneagent/)
[![CI](https://github.com/leelieber2025/AnnoTuneAgent/actions/workflows/tests.yml/badge.svg)](https://github.com/leelieber2025/AnnoTuneAgent/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/leelieber2025/AnnoTuneAgent/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24092828.svg)](https://doi.org/10.5281/zenodo.24092828)

AnnoTuneAgent annotates clustered scRNA-seq. `prepare` runs QC, gene
selection, and Leiden clustering. `run` collects three results for each
cluster: a marker score, a name-model result, and an ML-model result. A
judgment reads the three results and writes the title. A cluster can stay
unnamed. Clusters with the same title are merged.

The name model and the ML model are LoRA adapters fine-tuned on one shared
Qwen3.5-4B base. They run locally. Each copies a name from the candidates it is
shown:

- The **name model** copies from the marker dictionary hits
  (`dictionary_hits`) when that type's markers support the cluster.
- The **ML model** copies from measured candidates: differential expression
  and marker-panel statistics, including log fold change and the fraction of
  cells expressing each gene inside and outside the cluster.

Both models run when `models` is passed. Without it, `run` uses the marker
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
| {doc}`tutorials/pbmc8k_annotation` | PBMC 8k: `prepare`, then the name and ML models |
| {doc}`tutorials/load_model` | Download the weights and run a local or remote model server |
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
