# Tutorials

`prepare` is the first call. It prints the QC rules, then draws the QC
figures and the UMAP. `run` collects the marker score, the name-model result,
and the ML-model result, and writes the title. The model server stays loaded
between calls.

| Page | Contents |
|----------|----------|
| {doc}`pbmc8k_annotation` | Public PBMC 8k: `prepare`, then the name and ML models |
| {doc}`gse266598_annotation` | GEO series GSE266598: series record, download, `prepare`, annotation |
| {doc}`load_model` | Download the weights, then use a local server or connect to a remote one |
| {doc}`report` | What each part of the `run` report is for, including the text to paste |

{doc}`pbmc8k_annotation` uses the public 10x PBMC 8k count matrix (7,750 cells),
not the Scanpy PBMC 3k set. The matrix contains cells only, so AmbiDose is off.
The saved outputs come from this matrix.

```bash
pip install -e ".[dev,models]"
jupyter lab docs/tutorials/
```

`[models]` installs the tokenizer loader for local GGUF calls. Jupyter is not included; install it separately.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} PBMC 8k
:link: pbmc8k_annotation
:link-type: doc

`prepare`, then the name and ML models.
+++
Public PBMC 8k counts
:::

:::{grid-item-card} GEO series
:link: gse266598_annotation
:link-type: doc

From the GSE266598 page to a written title.
+++
Two mouse calvarial 10x samples
:::

:::{grid-item-card} Load a model
:link: load_model
:link-type: doc

Run a local server at `http://127.0.0.1:8080`, or connect to a remote one.
+++
Needs GGUF files or a running server
:::

:::{grid-item-card} Read the report
:link: report
:link-type: doc

The `Queries` block is text to paste into another model.
+++
No extra install
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

pbmc8k_annotation
gse266598_annotation
load_model
report
```
