# AnnoTuneAgent

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Cluster-level cell-type annotation for single-cell RNA-seq, with two
fine-tuned language models that run locally.**

Each cluster gets a marker score, a name-model result, and an ML-model
result. A rule-based judgment reads all three and writes the title, so every
call stays auditable.

## What is fine-tuned

One Qwen3.5-4B base (Q6_K GGUF) is shared by two independent LoRA adapters.
The adapters are not merged into the base, so they load together on a single
local `llama-server` and can be replaced without touching the base.

| Model | Task it was fine-tuned for | Input | Output |
| --- | --- | --- | --- |
| **name** | Choose a cell-type name from the marker dictionary | Cluster marker evidence and `dictionary_hits` | A name copied from `dictionary_hits` when that type's markers support the cluster |
| **ML** | Choose a name from measured cluster statistics | Differential expression, candidate marker panels, log fold change, fraction of cells expressing each gene inside and outside the cluster, panel summaries | A name copied from those candidates |

Both models are trained to **copy** a name from the candidates they are
shown, not to free-write one. A name outside the candidate list cannot become
a title, and a name with no marker that passes expression and specificity
validation is flagged (`no_marker_passed`) instead of being trusted silently.

The weights are on the Hugging Face Hub:
[lieberlee/annotune-qwen3.5-4b-gguf](https://huggingface.co/lieberlee/annotune-qwen3.5-4b-gguf)
(preview). Inference is local: no cluster data leaves the machine.

## Pipeline

```text
raw counts ─ prepare ─▶ QC, doublets, normalization, batch correction, Leiden
           ─ run ─────▶ per cluster:
                          1. marker score        (built-in human / mouse dictionary)
                          2. name model          (LoRA, copies from dictionary_hits)
                          3. ML model            (LoRA, copies from measured candidates)
                          ▼
                        judgment ─▶ title + status + reason + hint
                          ▼
                        validated marker panels, dotplot, report, annotated h5ad
```

- **Judgment, not voting.** Agreement among the three results gives a clear
  title. Disagreement, mixed programs, and unvalidated names are kept and
  flagged with a reason (`mixed_programs`, `near_clear_cluster`,
  `subcluster_labels`, `no_marker_passed`) so you can review them.
- **Marker evidence is checked, not quoted.** Panels are selected from cited
  dictionary markers and validated against measured expression and
  specificity in your data.
- **Every title is traceable.** The report records each model's answer, the
  panel, and the reason a cluster was or was not settled.

The models run when `models` is passed (or the CLI adapter flags are given).
Without them, `run` uses the marker score alone.

## Install

Python 3.10–3.13.

```bash
git clone https://github.com/leelieber2025/AnnoTuneAgent
cd AnnoTuneAgent
pip install -e .
```

Local GGUF models need `pip install -e '.[models]'`. Optional methods:
AmbiDose (`denoise=True`, [install](https://github.com/leelieber2025/AmbiDose)),
scVI and SOLO (`'.[scvi]'`), scFair (`'.[scfair]'`), Harmony, and Scanorama. If
you select one that is missing, `prepare` prints the install command and
leaves the matrix unchanged. See [Installation](docs/installation.md).

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
