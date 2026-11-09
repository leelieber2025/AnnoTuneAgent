# AnnoTuneAgent

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Cluster-level cell-type annotation for single-cell RNA-seq. `prepare` runs
QC, gene selection, and Leiden on integer counts. `run` scores a marker
dictionary and writes one name per cluster. An optional local model can
review a cluster when the markers do not settle it. The package does not
discover cell types that are absent from the dictionary.

## Install

Python 3.10–3.13.

```bash
git clone https://github.com/leelieber2025/AnnoTuneAgent
cd AnnoTuneAgent
pip install -e .
```

Ambient subtraction needs [AmbiDose](https://github.com/leelieber2025/AmbiDose)
and is off unless `prepare(..., denoise=True)`. scVI and SOLO need
`pip install -e '.[scvi]'`. scFair gene selection needs
`pip install -e '.[scfair]'`.

## First run

`prepare` expects raw integer counts. `run` expects finite, nonnegative,
library-size-normalized natural log1p expression, which `prepare` writes,
with Scanpy's `adata.uns["log1p"]`.

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

`--name-adapter`, `--ml-adapter`, and `--base` run the name and ML models on
one local server and require `--output-h5ad`. The weights are on the Hugging
Face Hub: [lieberlee/annotune-qwen3.5-4b-gguf](https://huggingface.co/lieberlee/annotune-qwen3.5-4b-gguf).
The tokenizer is installed with the package. See
[docs/tutorials/load_model.ipynb](docs/tutorials/load_model.ipynb).

## Status

**0.1.0 (Alpha).** Import as `import annotuneagent as ata`.

## License

Software: [Apache License 2.0](LICENSE).

## Author

**Zhao Li (李钊)**
Email: [leelieber@gmail.com](mailto:leelieber@gmail.com)
