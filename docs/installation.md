# Installation

Python 3.10–3.13. AnnData, Scanpy, NumPy, pandas, and SciPy are installed
with the package. Ambient subtraction needs AmbiDose, and only when you turn
it on. scFair is used only when `prepare(..., scfair=True)`.

| Package | Why it is there |
|---------|-----------------|
| `scanpy[leiden]>=1.10` | QC metrics, neighbors, Leiden `flavor="igraph"` |
| `scikit-misc>=0.1.4` | Default `seurat_v3` highly variable genes |
| `scfair` | Optional extra, `pip install annotuneagent[scfair]`. Variable genes and the resolution when `scfair=True` |
| `ambidose` | Optional ambient subtraction from a raw droplet matrix |
| `harmonypy` | Optional. Harmony neighbor-graph correction |
| `scanorama` | Optional. Scanorama neighbor-graph correction |
| `scvi-tools>=1.1.3` | Optional. scVI latent clustering and SOLO doublet detection; install with `pip install -e '.[scvi]'` |
| `igraph>=0.10` | Installed through Scanpy's Leiden extra |

```bash
pip install -e ".[dev]"
```

```python
import annotuneagent as ata
print(ata.__version__)
```

The version string is `src/annotuneagent/_version.py` (currently **0.1.0**).

GGUF weights are not included in the Python package. They are on the Hugging
Face Hub at [lieberlee/annotune-qwen3.5-4b-gguf](https://huggingface.co/lieberlee/annotune-qwen3.5-4b-gguf).
Download them as {doc}`tutorials/load_model` shows, then pass the directory to
`local_models(root)`. The Qwen tokenizer is installed with the package.
The Cell Ontology snapshot, reviewed name map, and granularity policy are also
installed with the package.
The included tokenizer files come from `Qwen/Qwen3.5-4B` under Apache-2.0;
their license is included at `annotuneagent/tokenizer/LICENSE`.

```bash
pip install -e ".[docs]"

# For local GGUF model calls
pip install -e ".[models]"
```

First call: {doc}`quickstart`. QC and annotation:
{doc}`tutorials/pbmc8k_annotation`. A local server or a remote one:
{doc}`tutorials/load_model`.
