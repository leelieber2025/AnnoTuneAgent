# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""AnnoTuneAgent public API.

Recommended usage::

    import annotuneagent as ata

    records = ata.annotate(adata, cluster_key="leiden", hypotheses=[...])
    adata = ata.read_10x(base_dir, ["sample_a", "sample_b"])
    info = ata.geo_info("GSE341410")
    adata = ata.read_geo(
        info, samples=["B1", "B3"], groups="B1: control B3: Day7",
        obs={"day": "B1: 0 B3: 7", "treatment": "B1: none B3: C&AIA"},
    )
    prepared = ata.prepare(adata, species="human", batch_key="sample")
    report = ata.run(
        prepared, "leiden", species="human",
        models=ata.ModelChain(
            name=ata.TaskEndpoint(base=model_file, maxlen=2048),
            ml=ata.TaskEndpoint(),
        ),
    )

``annotate`` scores caller-supplied hypotheses for each cluster. ``run``
collects the marker score and the name-model result. The check rule keeps
that name, replaces it, or drops it, then writes the title to
``annotune_label``. The title is a dictionary name. A cluster can stay
unnamed. Uncertain clusters are subclustered once. A cluster whose NK and T
panels are both specific is split even when the marker score named it, unless
the top panel is a specific mature NK T cell. The name model runs when a
``ModelChain`` is supplied. It copies a cell-type name from the marker
dictionary.
Pass the merged GGUF as ``TaskEndpoint(base=...)``. ``local_models`` still
reads an older directory that holds a base and a name adapter. The tokenizer
is installed with the package. The server is
``http://127.0.0.1:8080`` unless ``url`` or ``ANNOTUNE_URL`` says otherwise.
A matching server is reused. When the process list is unavailable, the check
reads ``/props`` and ``/lora-adapters``. ``TaskEndpoint(url=...)`` calls
``/completion`` on that server and enables the name adapter.
``attach`` connects to an existing service without loading weights.
Model requests wait 600 seconds unless ``timeout`` or ``ANNOTUNE_TIMEOUT``
says otherwise. ``ANNOTUNE_API_KEY`` is sent as a bearer token when the
server was started with ``--api-key``.
"""

from __future__ import annotations

from ._version import __version__
from .annotate import annotate
from .evaluate import evaluate
from .geo import GeoSample, GeoSeries, geo_info, read_geo
from .hypotheses import Hypothesis
from .local_lora import attach, local_server
from .markers import hypotheses_from_names, marker_names, suggest_tissues, tissue_pages
from .mltools import conformal_quantile, fit_ranker
from .prepare import prepare
from .read import read_10x
from .workflow import ModelChain, TaskEndpoint, dictionary_hypotheses, local_models, run

__all__ = [
    "GeoSample",
    "GeoSeries",
    "Hypothesis",
    "__version__",
    "annotate",
    "attach",
    "conformal_quantile",
    "dictionary_hypotheses",
    "evaluate",
    "fit_ranker",
    "geo_info",
    "hypotheses_from_names",
    "ModelChain",
    "TaskEndpoint",
    "local_models",
    "local_server",
    "marker_names",
    "prepare",
    "read_10x",
    "read_geo",
    "run",
    "suggest_tissues",
    "tissue_pages",
]
