from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .data import load_dataset, supported_dataset_names
from .preprocess import build_graphs
from .runner import MODEL_NAMES, evaluate_checkpoint, export_result_logs, run_contrastive_experiment





for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")


_MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "psgcl": {"dataset": "lgg", "epochs": 100},
    "without-contrastive": {"dataset": "gbm", "epochs": 50},
    "mcrgcn": {"dataset": "gbm", "epochs": 120},
    "psgcl-mcrgcn-graph": {"dataset": "gbm", "epochs": 100},
}
_COMMON_DEFAULTS: dict[str, Any] = {
    "data_dir": "dataset", "output_dir": "outputs", "folds": 10,
    "seeds": [223, 777, 2026], "device": "auto", "lr": 1e-3,
    "weight_decay": 5e-4, "hidden_dim": 256, "embedding_dim": 128,
    "dropout": 0.3, "graph_order": 2, "tau": 0.5, "lam": 0.5,
    "classifier_max_iter": 2000, "classifier_alpha": 0.001,
    "graph_source": None,
}


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}.")
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is not valid JSON: {exc.msg}.") from exc
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a JSON object.")
    return config


def _merge_train_args(parsed: argparse.Namespace) -> SimpleNamespace:
    cli_values = vars(parsed).copy()
    config = _load_config(cli_values.pop("config"))
    model = cli_values.get("model") or config.get("model") or "psgcl"
    if cli_values.get("ablation") is not None:
        if model != "psgcl" and model != cli_values["ablation"]:
            raise ValueError("--model and --ablation select conflicting experiments.")
        model = cli_values["ablation"]
    if model not in MODEL_NAMES:
        raise ValueError(f"--model must be one of: {', '.join(MODEL_NAMES)}.")
    allowed = set(_COMMON_DEFAULTS) | set(_MODEL_DEFAULTS[model]) | {
        "model", "run_name", "seed", "ablation", "command"
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError("Unknown configuration key(s): " + ", ".join(sorted(unknown)))
    resolved = {**_COMMON_DEFAULTS, **_MODEL_DEFAULTS[model], **config}
    for name, value in cli_values.items():
        if name in {"command", "ablation"}:
            continue
        if value is not None:
            resolved[name] = value
    if resolved.get("seed") is not None and cli_values.get("seeds") is not None:
        raise ValueError("Use either --seed or --seeds, not both.")
    if resolved.get("seed") is not None:
        resolved["seeds"] = [int(resolved["seed"])]
    resolved["model"] = model
    if int(resolved["epochs"]) < 1:
        raise ValueError("--epochs must be positive.")
    if float(resolved["lr"]) <= 0 or float(resolved["weight_decay"]) < 0:
        raise ValueError("--lr must be positive and --weight-decay must be non-negative.")
    if float(resolved["tau"]) <= 0 or not 0 <= float(resolved["lam"]) <= 1:
        raise ValueError("--tau must be positive and --lam must be in [0, 1].")
    if int(resolved["hidden_dim"]) < 1 or int(resolved["embedding_dim"]) < 1 or int(resolved["graph_order"]) < 1:
        raise ValueError("--hidden-dim, --embedding-dim, and --graph-order must be positive.")
    if not 0 <= float(resolved["dropout"]) < 1:
        raise ValueError("--dropout must be in [0, 1).")
    return SimpleNamespace(**resolved)


def _add_train_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=MODEL_NAMES, default=None)
    parser.add_argument(
        "--ablation",
        choices=("without-contrastive", "psgcl-mcrgcn-graph"),
        default=None,
    )
    parser.add_argument("--dataset", choices=supported_dataset_names(), default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--config", default=None, help="JSON configuration; explicit CLI values override it.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or a Torch device string.")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--graph-source",
        choices=("percentile", "legacy-mcrgcn"),
        default=None,
        help=(
            "Graph source for --model mcrgcn. The default is percentile; "
            "legacy-mcrgcn is supplied only for BRCA and GBM."
        ),
    )
    parser.add_argument("--graph-order", type=int, default=None, help="Bernstein propagation order K in P-SGCL.")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None, help="MCRGCN directional-loss coefficient.")
    parser.add_argument("--classifier-max-iter", type=int, default=None)
    parser.add_argument("--classifier-alpha", type=float, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible training, evaluation, preprocessing, and BRCA analysis for P-SGCL."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Run stratified CV for P-SGCL, MCRGCN, or implemented ablation.")
    _add_train_options(train)

    evaluate = commands.add_parser("evaluate", help="Reload a fold checkpoint and evaluate its saved hold-out split.")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--dataset", required=True, choices=supported_dataset_names())
    evaluate.add_argument("--data-dir", default="dataset")
    evaluate.add_argument("--device", default="auto")

    export_logs = commands.add_parser(
        "export-logs", help="Create readable per-seed logs for an existing completed central run."
    )
    export_logs.add_argument("--run-dir", required=True)

    inspect = commands.add_parser("inspect-data", help="Validate one supported MAT dataset and report its schema.")
    inspect.add_argument("--dataset", required=True, choices=supported_dataset_names())
    inspect.add_argument("--data-dir", default="dataset")

    preprocess = commands.add_parser(
        "preprocess", help="Build adaptive-percentile graphs without modifying supplied graphs."
    )
    preprocess.add_argument("--dataset", required=True, choices=supported_dataset_names())
    preprocess.add_argument("--data-dir", default="dataset")
    preprocess.add_argument("--output-dir", required=True)

    brca = commands.add_parser("brca-genes", help="Run BRCA-only Random-Forest/Enrichr interpretation.")
    brca.add_argument("args", nargs=argparse.REMAINDER, help="Pass options shown by python -m analysis.brca.run --help.")
    return parser


def _main(argv: list[str] | None = None, default_model: str | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if default_model is not None and (not arguments or arguments[0].startswith("-")):
        arguments = ["train", "--model", default_model, *arguments]
    parser = build_parser()
    parsed = parser.parse_args(arguments)
    if parsed.command == "train":
        run_dir = run_contrastive_experiment(_merge_train_args(parsed))
        print(f"Completed and reload-verified: {run_dir}")
    elif parsed.command == "evaluate":
        report = evaluate_checkpoint(parsed.checkpoint, parsed.dataset, parsed.data_dir, parsed.device)
        print(json.dumps(report, indent=2))
    elif parsed.command == "export-logs":
        run_dir = export_result_logs(parsed.run_dir)
        print(f"Created readable logs: {run_dir}")
    elif parsed.command == "inspect-data":
        dataset = load_dataset(parsed.dataset, parsed.data_dir)
        print(json.dumps({
            "dataset": dataset.name, "source": str(dataset.source_mat),
            "samples": dataset.n_samples, "classes": dataset.n_classes,
            "feature_counts": dataset.feature_counts,
            "raw_label_values": dataset.raw_label_values.tolist(),
            "edge_paths": [str(path) for path in dataset.edge_paths],
        }, indent=2))
    elif parsed.command == "preprocess":
        output = build_graphs(parsed.dataset, parsed.data_dir, parsed.output_dir)
        print(f"Created graph artifacts: {output}")
    elif parsed.command == "brca-genes":
        from analysis.brca.run import main as brca_main
        brca_main(parsed.args)


def main(argv: list[str] | None = None, default_model: str | None = None) -> None:
    try:
        _main(argv, default_model)
    except (FileNotFoundError, FileExistsError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
