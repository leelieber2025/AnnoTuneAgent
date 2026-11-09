# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""Read an h5ad, score clusters, write JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anndata as ad
import scanpy as sc

from .evaluate import evaluate
from .hypotheses import Hypothesis
from .local_lora import packaged_tokenizer
from .markers import hypotheses_from_names, parents_for, suggest_tissues, tissue_pages
from .prepare import install_hint, prepare
from .workflow import ModelChain, TaskEndpoint, dictionary_hypotheses, run


def _endpoint(
    url: str | None,
    adapter: str | None,
    base: str,
    maxlen: int,
    tokenizer: str | None,
    server: str | None,
) -> TaskEndpoint:
    return TaskEndpoint(
        url=url, adapter=adapter, base=base, maxlen=maxlen, tokenizer=tokenizer, server=server
    )


def _models(args, parser) -> ModelChain | None:
    specified = [
        args.name_url,
        args.ml_url,
        args.name_adapter,
        args.ml_adapter,
    ]
    if not any(specified):
        return None
    if not (args.name_url or args.name_adapter):
        parser.error("name needs --name-url or --name-adapter")
    if not (args.ml_url or args.ml_adapter):
        parser.error("ml needs --ml-url or --ml-adapter")
    if any((args.name_adapter, args.ml_adapter)):
        if args.name_url or args.ml_url:
            parser.error("local adapters cannot be combined with --name-url or --ml-url")
        if not (args.name_adapter and args.ml_adapter):
            parser.error("local adapters require both --name-adapter and --ml-adapter")
        if args.base is None:
            parser.error("local adapters require --base GGUF")
    return ModelChain(
        name=_endpoint(
            args.name_url,
            args.name_adapter,
            args.base or "",
            2048,
            packaged_tokenizer(),
            args.server,
        ),
        ml=_endpoint(
            args.ml_url, args.ml_adapter, args.base or "", 8192, packaged_tokenizer(), args.server
        ),
    )


def _hypotheses_from_file(path: Path) -> list[Hypothesis]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("hypotheses file must be a JSON list")
    hypotheses: list[Hypothesis] = []
    for item in payload:
        hypotheses.append(
            Hypothesis(
                str(item["name"]),
                positive=tuple(item["positive"]),
                negative=tuple(item.get("negative", ())),
            )
        )
    return hypotheses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="annotuneagent")
    parser.add_argument("--input", type=Path, help="count or clustered .h5ad")
    parser.add_argument(
        "--cluster-key", help="obs column of clusters; default leiden after --prepare"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON report; a .markers.pdf is written beside it when markers pass validation",
    )
    parser.add_argument(
        "--prepare", action="store_true", help="filter, select genes, and cluster before annotation"
    )
    parser.add_argument(
        "--batch-key", help="obs column for per-sample QC, HVG, and ambient subtraction"
    )
    parser.add_argument(
        "--batch-correction",
        choices=["none", "harmony", "combat", "scanorama", "scvi", "scvi+harmony"],
        default="none",
        help=(
            "neighbor representation for Leiden and UMAP; "
            "scvi and scvi+harmony use raw counts and can use GPU"
        ),
    )
    parser.add_argument("--doublet-method", choices=["none", "scrublet", "solo"], default="none")
    parser.add_argument(
        "--doublet-chip",
        choices=["10k", "20k"],
        default="10k",
        help=(
            "multiplet rate for Scrublet and SOLO from recovered cells: "
            "10k is 0.8%% per 1,000, 20k is 0.4%% per 1,000"
        ),
    )
    parser.add_argument(
        "--solo-expected-rates",
        type=Path,
        help="JSON map of sample to expected doublet fraction; requires --doublet-method solo",
    )
    parser.add_argument(
        "--scvi-model-dir", type=Path, help="save the trained scVI integration model"
    )
    parser.add_argument(
        "--denoise", action="store_true", help="subtract ambient RNA with AmbiDose; requires --raw"
    )
    parser.add_argument(
        "--raw",
        type=Path,
        help=(
            "unfiltered counts for AmbiDose: one matrix, or a parent folder "
            "with one sample subfolder each (raw_feature_bc_matrix.h5)"
        ),
    )
    parser.add_argument("--resolution", type=float, help="Leiden resolution; default 0.4")
    parser.add_argument(
        "--n-pcs", type=int, help="principal components; default is the variance elbow"
    )
    parser.add_argument("--n-top-genes", type=int, help="highly variable genes; default 2000")
    parser.add_argument(
        "--norm",
        choices=["log1p", "shifted"],
        default="log1p",
        help="log1p scales to 10000 counts; shifted scales to the median library size",
    )
    parser.add_argument(
        "--scfair", action="store_true", help="use scFair for variable genes and the resolution"
    )
    parser.add_argument(
        "--max-pct-mt",
        type=float,
        help="mitochondrial percent cap from 0 to 100; 10 means 10%%",
    )
    parser.add_argument("--output-h5ad", type=Path, help="write workflow labels to this .h5ad")
    parser.add_argument(
        "--types",
        action="append",
        help="built-in dictionary name; repeat for each candidate",
    )
    parser.add_argument("--hypotheses", type=Path, help="JSON list of {name, positive, negative}")
    parser.add_argument(
        "--catalog",
        type=Path,
        help="full candidate pool the model may nominate; defaults to the species dictionary",
    )
    parser.add_argument("--species", choices=["human", "mouse"], help="required with --types")
    parser.add_argument("--tissue", help="adds that tissue page to the core dictionary")
    parser.add_argument(
        "--list-tissues",
        action="store_true",
        help="print tissue pages and exit; with --author-tissue, print the matching pages",
    )
    parser.add_argument(
        "--author-tissue",
        help="author tissue phrase, such as DRG; choose one printed page with --tissue",
    )
    parser.add_argument("--label-key", help="author labels in obs, used only for evaluation")
    parser.add_argument("--min-margin", type=float, default=None)
    parser.add_argument(
        "--server",
        help="server root, default http://127.0.0.1:8080; reuse requires matching GGUF and adapters",
    )
    parser.add_argument("--name-url", help="name adapter server root")
    parser.add_argument("--ml-url", help="ml adapter server root")
    parser.add_argument("--name-adapter", help="name LoRA GGUF; loaded with ml on one base")
    parser.add_argument("--ml-adapter", help="ml LoRA GGUF; enabled only for the ml request")
    parser.add_argument("--base", help="GGUF base model for local adapters")
    args = parser.parse_args(argv)
    if args.list_tissues:
        pages = (
            suggest_tissues(args.author_tissue, args.species)
            if args.author_tissue
            else tissue_pages(args.species)
        )
        sys.stdout.write("\n".join(pages) + ("\n" if pages else ""))
        return 0
    if args.input is None or args.output is None:
        parser.error("--input and --output are required")
    if args.tissue is not None:
        if args.species is None:
            parser.error("--tissue requires --species")
        if args.tissue not in tissue_pages(args.species):
            parser.error(
                f"unknown {args.species} tissue {args.tissue!r}; "
                f"choose one from --list-tissues --species {args.species}"
            )
    if not args.prepare:
        prepare_options = [
            name
            for name, used in (
                ("--batch-key", args.batch_key is not None),
                ("--batch-correction", args.batch_correction != "none"),
                ("--doublet-method", args.doublet_method != "none"),
                ("--doublet-chip", args.doublet_chip != "10k"),
                ("--solo-expected-rates", args.solo_expected_rates is not None),
                ("--scvi-model-dir", args.scvi_model_dir is not None),
                ("--denoise", args.denoise),
                ("--raw", args.raw is not None),
                ("--resolution", args.resolution is not None),
                ("--n-pcs", args.n_pcs is not None),
                ("--n-top-genes", args.n_top_genes is not None),
                ("--norm", args.norm != "log1p"),
                ("--scfair", args.scfair),
                ("--max-pct-mt", args.max_pct_mt is not None),
            )
            if used
        ]
        if prepare_options:
            parser.error(f"{', '.join(prepare_options)} require --prepare")
    if args.raw is not None and not args.denoise:
        parser.error("--raw requires --denoise")
    if args.author_tissue and args.tissue is None:
        pages = suggest_tissues(args.author_tissue, args.species)
        parser.error("choose one tissue page with --tissue: " + ", ".join(pages))
    if args.author_tissue and args.tissue is not None:
        pages = suggest_tissues(args.author_tissue, args.species)
        full = tissue_pages(args.species)
        if pages != full and args.tissue not in pages:
            print(
                f"author source {args.author_tissue!r} matches {', '.join(pages)}; "
                f"this run uses --tissue {args.tissue!r}",
                file=sys.stderr,
            )

    if (args.types is None) == (args.hypotheses is None):
        parser.error("pass exactly one of --types or --hypotheses")
    models = _models(args, parser)
    if models is not None and args.output_h5ad is None:
        parser.error("model endpoints require --output-h5ad")
    # Fail now. Prepare and the models can run for a long time before anything is written.
    for path in (args.output, args.output_h5ad):
        if path is None:
            continue
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            parser.error(f"cannot create {path.parent}: {exc}")
        if not os.access(path.parent, os.W_OK):
            parser.error(f"cannot write to {path.parent}")

    if args.types is not None:
        if args.species is None:
            parser.error("--species is required with --types")
        names = [part.strip() for part in args.types if part.strip()]
        hypotheses = hypotheses_from_names(names, args.species, args.tissue)
        parents = parents_for(names, species=args.species, tissue=args.tissue)
    else:
        hypotheses = _hypotheses_from_file(args.hypotheses)
        parents = {}
    if args.catalog is not None:
        catalog = _hypotheses_from_file(args.catalog)
    elif args.species is not None:
        catalog = dictionary_hypotheses(args.species, args.tissue)
    else:
        catalog = None

    adata = sc.read_h5ad(args.input)
    if args.prepare:
        if args.species is None:
            parser.error("--prepare requires --species")
        if args.denoise and args.raw is None:
            parser.error("No unfiltered droplet matrix. Pass --raw, or omit --denoise.")
        if install_hint(
            batch_correction=args.batch_correction,
            doublet_method=args.doublet_method,
            scfair=args.scfair,
            denoise=args.denoise,
        ):
            return 0
        rates = None
        if args.solo_expected_rates is not None:
            rates = json.loads(args.solo_expected_rates.read_text())
            if not isinstance(rates, dict):
                parser.error("--solo-expected-rates must contain a JSON object")
        adata = prepare(
            adata,
            species=args.species,
            batch_key=args.batch_key,
            batch_correction=args.batch_correction,
            doublet_method=args.doublet_method,
            doublet_chip=args.doublet_chip,
            solo_expected_rates=rates,
            scvi_model_dir=args.scvi_model_dir,
            denoise=args.denoise,
            raw=args.raw,
            resolution=args.resolution,
            n_pcs=args.n_pcs,
            n_top_genes=args.n_top_genes,
            norm=args.norm,
            scfair=args.scfair,
            max_pct_mt=args.max_pct_mt,
            cluster_key=args.cluster_key or "leiden",
        )
        cluster_key = args.cluster_key or "leiden"
    elif args.cluster_key is None:
        parser.error("--cluster-key is required")
    else:
        cluster_key = args.cluster_key
    kwargs = {}
    if args.min_margin is not None:
        kwargs["min_margin"] = args.min_margin
    payload = run(
        adata,
        cluster_key,
        species=args.species,
        tissue=args.tissue,
        hypotheses=hypotheses,
        catalog=catalog,
        models=models,
        source=args.input.name,
        dotplot_path=args.output.with_name(args.output.stem + ".markers.pdf"),
        **kwargs,
    )
    if args.label_key is not None:
        payload["evaluation"] = evaluate(
            payload["records"],
            adata,
            cluster_key,
            args.label_key,
            parents=parents,
            prediction_key="annotune_label",
        )
    # The report goes first so a failed h5ad write does not lose the run.
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.output_h5ad is not None:
        saved = adata.copy()
        if "annotune_prep" in saved.uns:
            saved.uns["annotune_prep"] = json.dumps(saved.uns["annotune_prep"])
        with ad.settings.override(allow_write_nullable_strings=True):
            saved.write_h5ad(args.output_h5ad)
    return 0
