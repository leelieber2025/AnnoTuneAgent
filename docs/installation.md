# Installation

Python 3.10–3.13.

```bash
pip install annotuneagent
```

```python
import annotuneagent as ata
print(ata.__version__)
```

The version is set in `src/annotuneagent/_version.py` (currently **0.1.2**).

From a checkout:

```bash
git clone https://github.com/leelieber2025/AnnoTuneAgent.git
cd AnnoTuneAgent
pip install -e ".[dev]"
pip install -e ".[docs]"
```

## Dependencies

AnnData, Scanpy, NumPy, pandas, SciPy, and scikit-learn are installed with the
package. These are optional:

| Package | Purpose | Install |
|---------|---------|---------|
| `ambidose` | Ambient RNA subtraction (`denoise=True`) | see [AmbiDose](https://github.com/leelieber2025/AmbiDose) |
| `harmonypy` | Harmony batch correction | `pip install harmonypy` |
| `scanorama` | Scanorama batch correction | `pip install scanorama` |
| `scvi-tools>=1.1.3` | scVI batch correction and SOLO doublets | `pip install 'annotuneagent[scvi]'` |
| `scfair>=0.8` | scFair gene selection (`scfair=True`) | `pip install 'annotuneagent[scfair]'` |
| `transformers` | Tokenizer loader for local GGUF models | `pip install 'annotuneagent[models]'` |

If you select an optional method and its package is missing, `prepare` prints
the install command and leaves the matrix unchanged. ComBat is part of Scanpy.

## Model weights

The GGUF weights are not in the Python package. Download them from the
Hugging Face Hub:
[lieberlee/annotune-qwen3.5-4b-gguf](https://huggingface.co/lieberlee/annotune-qwen3.5-4b-gguf).
{doc}`tutorials/load_model` shows the download and passes the folder to
`local_models(root)`.

The package includes the Qwen tokenizer files, the Cell Ontology snapshot, and
the reviewed name map. The tokenizer files come from `Qwen/Qwen3.5-4B`
(Apache-2.0); the license is at `annotuneagent/tokenizer/LICENSE`.

## Next

{doc}`quickstart` for a first call, {doc}`tutorials/pbmc8k_annotation` for QC and
annotation, {doc}`tutorials/gse266598_annotation` for a GEO series from the
series page through annotation, and {doc}`tutorials/load_model` for a local or
remote model server.
