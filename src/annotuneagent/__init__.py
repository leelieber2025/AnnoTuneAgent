# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""AnnoTuneAgent public API.

Recommended usage::

    import annotuneagent as ata

    records = ata.annotate(adata, cluster_key="leiden", hypotheses=[...])
    adata = ata.read_10x(base_dir, ["sample_a", "sample_b"])
    prepared = ata.prepare(adata, species="human", batch_key="sample")
    report = ata.run(
        prepared, "leiden", species="human",
        models=ata.local_models(model_dir),
    )

``annotate`` scores caller-supplied hypotheses for each cluster. ``run``
collects three results, the marker score, the name model, and the ML model,
then a judgment writes the title to ``annotune_label``. The title is a
dictionary name. A cluster can stay unnamed. Uncertain clusters are
subclustered once. The name and ML models run when a ``ModelChain`` is
supplied. The name model copies a cell-type name from the marker dictionary.
The ML model is trained on machine-learning measurements of the cluster,
differential expression and marker-panel statistics, and copies a name from
those candidates.
``local_models`` takes the directory containing the Q6_K base and the name and
ml adapters. The tokenizer is installed with the package. The server is
``http://127.0.0.1:8080`` unless ``url`` or ``ANNOTUNE_URL`` says otherwise.
A matching server is reused. When the process list is unavailable, the check
reads ``/props`` and ``/lora-adapters``. ``TaskEndpoint(url=...)`` calls
``/completion`` on that server and enables one adapter per request.
``attach`` connects to an existing service without loading weights.
Model requests wait 600 seconds unless ``timeout`` or ``ANNOTUNE_TIMEOUT``
says otherwise.
"""

from __future__ import annotations

from ._version import __version__
from .annotate import annotate
from .evaluate import evaluate
from .hypotheses import Hypothesis
from .local_lora import attach, local_server
from .markers import hypotheses_from_names, marker_names, suggest_tissues, tissue_pages
from .mltools import conformal_quantile, fit_ranker
from .prepare import prepare
from .read import read_10x
from .workflow import ModelChain, TaskEndpoint, dictionary_hypotheses, local_models, run

__all__ = [
    "Hypothesis",
    "__version__",
    "annotate",
    "attach",
    "conformal_quantile",
    "dictionary_hypotheses",
    "evaluate",
    "fit_ranker",
    "hypotheses_from_names",
    "ModelChain",
    "TaskEndpoint",
    "local_models",
    "local_server",
    "marker_names",
    "prepare",
    "read_10x",
    "run",
    "suggest_tissues",
    "tissue_pages",
]
