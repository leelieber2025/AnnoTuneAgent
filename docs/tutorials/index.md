# Tutorials

`prepare` is the first call. It prints the QC rules, then draws the QC
figures and the UMAP. The second call is annotation with the name and ml
adapters. The server stays loaded.

| Notebook | What it does |
|----------|----------------|
| {doc}`pbmc8k_annotation` | Public PBMC 8k. `prepare`, then name and ml. |
| {doc}`load_model` | Use a local server, or connect to one that is already running. |

{doc}`pbmc8k_annotation` loads the public PBMC 8k count matrix (7,750 cells),
not the Scanpy PBMC 3k set. That file is already cells, so AmbiDose stays off.
The saved outputs are from this matrix.

```bash
pip install -e ".[dev,models]"
jupyter lab docs/tutorials/
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} PBMC 8k
:link: pbmc8k_annotation
:link-type: doc

`prepare`, then name and ml. The server stays loaded.
+++
Public PBMC 8k counts
:::

:::{grid-item-card} Load a model
:link: load_model
:link-type: doc

Use a local server at `http://127.0.0.1:8080`, or connect to a remote one.
+++
Your server or GGUF files; the tokenizer is installed with the package
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

pbmc8k_annotation
load_model
```
