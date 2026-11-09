# AnnoTuneAgent Documentation

AnnoTuneAgent annotates clustered scRNA-seq. `prepare` runs QC, gene
selection, and Leiden. `run` scores the marker dictionary and writes one
name per cluster. Clusters that receive the same name are merged.

```python
import annotuneagent as ata

adata = ata.prepare(adata, species="human")
report = ata.run(adata, "leiden", species="human", subcluster=False)
```

`prepare` expects raw integer counts. AmbiDose ambient subtraction stays
off unless you pass `denoise=True` and the unfiltered droplet matrix.

| If you need | Pass |
|-------------|------|
| A sample column | `batch_key="sample"` |
| Doublet removal | `doublet_method="scrublet"` or `"solo"` |
| 10k or 20k chip | `doublet_chip="10k"` (default) or `"20k"` |
| Per-sample SOLO removal estimates | `solo_expected_rates={"sample_a": 0.08, "sample_b": 0.06}` |
| scVI latent clustering | `batch_correction="scvi"` |
| scVI then Harmony | `batch_correction="scvi+harmony"` |
| Ambient subtraction | `denoise=True` and per-sample `raw`: a parent folder, or a dict keyed by sample |
| A different Leiden resolution | `resolution=0.8` |
| A different HVG count | `n_top_genes=3000` |
| scFair gene selection and resolution | `scfair=True` |

The default selects 2000 genes with Scanpy `seurat_v3` and clusters at
resolution 0.4. A caller-supplied resolution is used as given. `scfair=True`
uses scFair for the gene list and sets the resolution to half of scFair's
suggestion.

| Page | Contents |
|------|----------|
| {doc}`installation` · {doc}`quickstart` | Install and a first call |
| {doc}`tutorials/pbmc8k_annotation` | PBMC 8k: prepare, then name and ml |
| {doc}`tutorials/load_model` | Local server, or a remote one that is already running |
| {doc}`api/index` | Signatures |
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
