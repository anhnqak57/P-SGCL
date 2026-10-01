from __future__ import annotations

import json
import pickle
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from baselines.mcrgcn import Contrast, HeCo
from models.psgcl import MultiContrastiveLoss, PSGCL, PSGCLWithoutContrastive

from .data import (
    OmicsDataset,
    graph_edge_paths,
    load_all_edge_indexes,
    load_dataset,
    resolve_device,
    validate_folds,
)
from .metrics import METRIC_NAMES, classification_metrics, summarise


MODEL_NAMES = ("psgcl", "without-contrastive", "mcrgcn", "psgcl-mcrgcn-graph")
MCRGCN_GRAPH_ABLATION_DATASETS = ("brca", "gbm")


def graph_source_for_model(model_name: str) -> str:
    if model_name == "psgcl-mcrgcn-graph":
        return "legacy-mcrgcn"
    return "percentile"


def resolve_graph_source(model_name: str, requested_source: str | None) -> str:
    if model_name == "mcrgcn":
        return requested_source or "percentile"
    if requested_source is not None:
        raise ValueError(
            "--graph-source is supported only with --model mcrgcn. "
            "Use --model psgcl-mcrgcn-graph for the P-SGCL graph ablation."
        )
    return graph_source_for_model(model_name)


def validate_model_dataset(model_name: str, dataset: OmicsDataset) -> None:
    if model_name == "psgcl-mcrgcn-graph":
        if dataset.name not in MCRGCN_GRAPH_ABLATION_DATASETS or dataset.n_classes != 4:
            supported = ", ".join(MCRGCN_GRAPH_ABLATION_DATASETS)
            raise ValueError(
                "psgcl-mcrgcn-graph is supported only for the supplied four-class "
                f"{supported.upper()} datasets; received {dataset.name!r} with "
                f"{dataset.n_classes} classes."
            )


@dataclass
class FoldArtifacts:
    fold: int
    seed: int
    checkpoint: str
    classifier: str | None
    metrics: dict[str, float]
    reload_verified: bool


def _format_metric(value: float) -> str:
    return "NaN" if not np.isfinite(value) else f"{value:.6f}"


def _format_fold_report(artifact: FoldArtifacts, total_folds: int) -> str:
    metric_lines = "  ".join(
        f"{name}={_format_metric(artifact.metrics[name])}" for name in METRIC_NAMES
    )
    classifier = artifact.classifier or "not applicable"
    return "\n".join(
        (
            f"Completed seed={artifact.seed}, fold={artifact.fold}/{total_folds}",
            f"  {metric_lines}",
            f"  checkpoint={artifact.checkpoint}",
            f"  classifier={classifier}",
            f"  reload_verified={artifact.reload_verified}",
        )
    )


def _format_summary_report(
    summary: dict[str, dict[str, float]],
    title: str = "Cross-validation summary (mean +/- std)",
) -> str:
    lines = [f"{title}:"]
    lines.extend(
        f"  {name}={_format_metric(summary[name]['mean'])} +/- {_format_metric(summary[name]['std'])}"
        for name in METRIC_NAMES
    )
    return "\n".join(lines)


def _summarise_seed_results(folds: list[FoldArtifacts]) -> list[dict[str, Any]]:
    metrics_by_seed: dict[int, list[dict[str, float]]] = {}
    for artifact in folds:
        metrics_by_seed.setdefault(artifact.seed, []).append(artifact.metrics)
    return [
        {"seed": seed, "summary": summarise(metrics)}
        for seed, metrics in metrics_by_seed.items()
    ]


def _summarise_seed_means(seed_summaries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return summarise(
        [
            {name: float(item["summary"][name]["mean"]) for name in METRIC_NAMES}
            for item in seed_summaries
        ]
    )


def _write_log(path: Path, report: str, *, echo: bool) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(report)
        handle.write("\n\n")
    if echo:
        print(report, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialise {type(value).__name__}.")


def _make_run_dir(output_dir: str | Path, dataset: str, model: str, run_name: str | None) -> Path:
    base = Path(output_dir).resolve() / dataset / model
    name = run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = base / name
    if destination.exists():
        raise FileExistsError(
            f"Output directory {destination} already exists. Choose --run-name or a new --output-dir; "
            "existing research outputs are never overwritten."
        )
    destination.mkdir(parents=True)
    return destination


def _scaled_graph_data(
    dataset: OmicsDataset,
    edge_indexes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    train_idx: np.ndarray,
    device: torch.device,
    saved_scalers: list[dict[str, list[float]]] | None = None,
) -> tuple[tuple[Data, Data, Data], list[dict[str, list[float]]]]:
    transformed: list[np.ndarray] = []
    scaler_state: list[dict[str, list[float]]] = []
    for position, view in enumerate(dataset.views):
        if saved_scalers is None:
            scaler = StandardScaler().fit(view[train_idx])
            state = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
        else:
            state = saved_scalers[position]
            scaler = StandardScaler()
            scaler.mean_ = np.asarray(state["mean"], dtype=np.float64)
            scaler.scale_ = np.asarray(state["scale"], dtype=np.float64)
            scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = scaler.mean_.size
        transformed.append(scaler.transform(view).astype(np.float32, copy=False))
        scaler_state.append(state)

    labels = torch.as_tensor(dataset.labels, dtype=torch.long)
    data_objects = tuple(
        Data(
            x=torch.as_tensor(view, dtype=torch.float32),
            edge_index=edge_index,
            y=labels,
        ).to(device)
        for view, edge_index in zip(transformed, edge_indexes)
    )
    return data_objects, scaler_state


def _model_and_loss(
    model_name: str,
    feature_dims: tuple[int, int, int],
    n_classes: int,
    args: SimpleNamespace,
    device: torch.device,
) -> tuple[nn.Module, nn.Module | None, dict[str, Any]]:
    common = {
        "hidden_dim": int(args.hidden_dim),
        "output_dim": int(args.embedding_dim),
        "dropout": float(args.dropout),
    }
    if model_name in {"psgcl", "psgcl-mcrgcn-graph"}:
        kwargs = {**common, "K": int(args.graph_order)}
        model = PSGCL(*feature_dims, **kwargs).to(device)
        return model, MultiContrastiveLoss(tau=float(args.tau)).to(device), kwargs
    if model_name == "mcrgcn":
        model = HeCo(*feature_dims, **common).to(device)
        criterion = Contrast(int(args.embedding_dim), tau=float(args.tau), lam=float(args.lam)).to(device)
        return model, criterion, common
    if model_name == "without-contrastive":
        kwargs = {**common, "num_classes": n_classes}
        model = PSGCLWithoutContrastive(*feature_dims, **kwargs).to(device)
        return model, nn.CrossEntropyLoss().to(device), kwargs
    raise ValueError(f"Unsupported runner model {model_name!r}.")


def _embeddings(model_name: str, model: nn.Module, graph_data: tuple[Data, Data, Data]) -> np.ndarray:
    if model_name == "mcrgcn":
        return model.get_embeds(*graph_data)
    return model.get_embeds(*graph_data)


def _reload_model(
    checkpoint: Path,
    graph_data: tuple[Data, Data, Data],
    device: torch.device,
) -> np.ndarray:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model_name = payload["model_name"]
    feature_dims = tuple(payload["feature_dims"])
    kwargs = payload["model_kwargs"]
    if model_name in {"psgcl", "psgcl-mcrgcn-graph"}:
        model = PSGCL(*feature_dims, **kwargs)
    elif model_name == "mcrgcn":
        model = HeCo(*feature_dims, **kwargs)
    elif model_name == "without-contrastive":
        model = PSGCLWithoutContrastive(*feature_dims, **kwargs)
    else:
        raise ValueError(f"Checkpoint names unknown model {model_name!r}.")
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    with torch.no_grad():
        if model_name == "without-contrastive":
            result = model.get_embeds(*graph_data)
        else:
            result = _embeddings(model_name, model, graph_data)
    if not np.isfinite(result).all():
        raise RuntimeError(f"Reloaded checkpoint {checkpoint} produced a non-finite embedding.")
    return result


def _save_checkpoint(
    path: Path,
    model_name: str,
    model: nn.Module,
    model_kwargs: dict[str, Any],
    feature_dims: tuple[int, int, int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    scaler_state: list[dict[str, list[float]]],
    seed: int,
    graph_source: str,
) -> None:
    torch.save(
        {
            "format_version": 2,
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "feature_dims": feature_dims,
            "model_state": model.state_dict(),
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "scalers": scaler_state,
            "seed": seed,
            "graph_source": graph_source,
        },
        path,
    )


def run_contrastive_experiment(args: SimpleNamespace) -> Path:
    if args.model not in MODEL_NAMES:
        raise ValueError(f"This runner supports {', '.join(MODEL_NAMES)}.")
    dataset = load_dataset(args.dataset, args.data_dir)
    validate_model_dataset(args.model, dataset)
    validate_folds(dataset.labels, int(args.folds))
    device = resolve_device(args.device)
    graph_source = resolve_graph_source(args.model, args.graph_source)
    edge_paths = graph_edge_paths(dataset, graph_source)
    edge_indexes = load_all_edge_indexes(dataset, graph_source)
    run_dir = _make_run_dir(args.output_dir, dataset.name, args.model, args.run_name)
    (run_dir / "checkpoints").mkdir()
    seed_log_dir = run_dir / "seed_logs"
    seed_log_dir.mkdir()
    run_log = run_dir / "training.log"

    config = vars(args).copy()
    config.update(
        {
            "resolved_dataset": dataset.name,
            "source_mat": str(dataset.source_mat),
            "n_samples": dataset.n_samples,
            "n_classes": dataset.n_classes,
            "feature_counts": dataset.feature_counts,
            "raw_label_values": dataset.raw_label_values.tolist(),
            "graph_source": graph_source,
            "edge_paths": [str(path) for path in edge_paths],
            "resolved_device": str(device),
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=_json_default), encoding="utf-8")
    _write_log(
        run_log,
        "\n".join(
            (
                "Central training started",
                f"model={args.model}  dataset={dataset.name}  device={device}",
                f"seeds={list(args.seeds)}  folds={args.folds}  epochs={args.epochs}",
                "Per-seed logs: seed_logs/seed_<seed>.log",
            )
        ),
        echo=False,
    )

    all_folds: list[FoldArtifacts] = []
    seed_summaries: list[dict[str, Any]] = []
    seeds = tuple(args.seeds)
    for seed in seeds:
        set_seed(int(seed))
        seed_log = seed_log_dir / f"seed_{seed}.log"
        _write_log(seed_log, f"Results for seed={seed}", echo=False)
        seed_folds: list[FoldArtifacts] = []
        splitter = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(seed))
        for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(dataset.n_samples), dataset.labels), start=1):
            graph_data, scaler_state = _scaled_graph_data(dataset, edge_indexes, train_idx, device)
            model, criterion, model_kwargs = _model_and_loss(
                args.model, dataset.feature_counts, dataset.n_classes, args, device
            )
            optimizer = torch.optim.Adam(
                model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
            )
            model.train()
            for _ in range(int(args.epochs)):
                optimizer.zero_grad(set_to_none=True)
                if args.model == "without-contrastive":
                    logits = model(*graph_data)
                    loss = criterion(logits[train_idx], graph_data[0].y[train_idx])
                else:
                    representations = model(*graph_data)
                    y_train = graph_data[0].y[train_idx]
                    positives = (y_train[:, None] == y_train[None, :]).float()
                    loss = criterion(*(representation[train_idx] for representation in representations), positives)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss in seed {seed}, fold {fold}.")
                loss.backward()
                if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                    raise RuntimeError(f"Non-finite gradient in seed {seed}, fold {fold}.")
                optimizer.step()

            model.eval()
            classifier_path: Path | None = None
            with torch.no_grad():
                embeddings = _embeddings(args.model, model, graph_data)
                if args.model == "without-contrastive":
                    probability = torch.softmax(model(*graph_data), dim=1).cpu().numpy()
                    prediction = probability.argmax(axis=1)
                else:
                    classifier = MLPClassifier(
                        activation="tanh",
                        max_iter=int(args.classifier_max_iter),
                        solver="adam",
                        alpha=float(args.classifier_alpha),
                        hidden_layer_sizes=(60, 30),
                        random_state=int(seed),
                    )
                    classifier.fit(embeddings[train_idx], dataset.labels[train_idx])
                    prediction = classifier.predict(embeddings)
                    probability = classifier.predict_proba(embeddings)
                    classifier_path = run_dir / "checkpoints" / f"seed_{seed}_fold_{fold}_classifier.pkl"
                    with classifier_path.open("wb") as handle:
                        pickle.dump(classifier, handle)

            metric_values = classification_metrics(
                dataset.labels[test_idx], prediction[test_idx], probability[test_idx],
                embeddings[test_idx], dataset.n_classes,
            )
            checkpoint = run_dir / "checkpoints" / f"seed_{seed}_fold_{fold}.pt"
            _save_checkpoint(
                checkpoint, args.model, model, model_kwargs, dataset.feature_counts,
                train_idx, test_idx, scaler_state, int(seed), graph_source,
            )
            reloaded_embeddings = _reload_model(checkpoint, graph_data, device)
            if reloaded_embeddings.shape != embeddings.shape or not np.allclose(
                reloaded_embeddings, embeddings, rtol=1e-5, atol=1e-6
            ):
                raise RuntimeError(f"Checkpoint reload changed embeddings for seed {seed}, fold {fold}.")
            artifact = FoldArtifacts(
                fold=fold,
                seed=int(seed),
                checkpoint=str(checkpoint.relative_to(run_dir)),
                classifier=(str(classifier_path.relative_to(run_dir)) if classifier_path else None),
                metrics=metric_values,
                reload_verified=True,
            )
            all_folds.append(artifact)
            seed_folds.append(artifact)
            fold_report = _format_fold_report(artifact, int(args.folds))
            _write_log(run_log, fold_report, echo=True)
            _write_log(seed_log, fold_report, echo=False)

        seed_summary = summarise([item.metrics for item in seed_folds])
        seed_summaries.append({"seed": int(seed), "summary": seed_summary})
        seed_report = _format_summary_report(
            seed_summary,
            f"Seed {seed} summary across {len(seed_folds)} folds (mean +/- std)",
        )
        _write_log(run_log, seed_report, echo=True)
        _write_log(seed_log, seed_report, echo=False)

    fold_summary = summarise([item.metrics for item in all_folds])
    seed_mean_summary = _summarise_seed_means(seed_summaries)
    report = {
        "folds": [asdict(item) for item in all_folds],
        "summary": fold_summary,
        "seed_summaries": seed_summaries,
        "seed_mean_summary": seed_mean_summary,
    }
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    _write_log(
        run_log,
        _format_summary_report(
            seed_mean_summary,
            f"Final summary across {len(seed_summaries)} seed means (mean +/- std)",
        ),
        echo=True,
    )
    return run_dir


def export_result_logs(run_dir: str | Path, *, echo: bool = True) -> Path:
    directory = Path(run_dir).resolve()
    metrics_path = directory / "metrics.json"
    config_path = directory / "config.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics report not found: {metrics_path}.")
    if not config_path.is_file():
        raise FileNotFoundError(f"Run configuration not found: {config_path}.")
    run_log = directory / "training.log"
    seed_log_dir = directory / "seed_logs"
    if seed_log_dir.exists():
        raise FileExistsError(
            f"Per-seed logs already exist in {directory}; they are never overwritten."
        )

    try:
        stored_config = json.loads(config_path.read_text(encoding="utf-8"))
        stored_report = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Run result files in {directory} are not valid JSON: {exc.msg}.") from exc
    stored_folds = stored_report.get("folds")
    summary = stored_report.get("summary")
    if not isinstance(stored_folds, list) or not stored_folds:
        raise ValueError(f"{metrics_path} must contain a non-empty 'folds' list.")
    if not isinstance(summary, dict):
        raise ValueError(f"{metrics_path} must contain a 'summary' object.")

    seed_log_dir.mkdir()
    total_folds = int(stored_config.get("folds", max(int(item["fold"]) for item in stored_folds)))
    write_run_log = not run_log.exists()
    if write_run_log:
        _write_log(
            run_log,
            "\n".join(
                (
                    "Central results exported from metrics.json",
                    f"model={stored_config.get('model', 'unknown')}  dataset={stored_config.get('resolved_dataset', stored_config.get('dataset', 'unknown'))}",
                    f"folds={total_folds}",
                    "Per-seed logs: seed_logs/seed_<seed>.log",
                )
            ),
            echo=False,
        )
    exported_seeds: set[int] = set()
    exported_folds: list[FoldArtifacts] = []
    for item in stored_folds:
        try:
            artifact = FoldArtifacts(
                fold=int(item["fold"]),
                seed=int(item["seed"]),
                checkpoint=str(item["checkpoint"]),
                classifier=(str(item["classifier"]) if item.get("classifier") else None),
                metrics={name: float(item["metrics"][name]) for name in METRIC_NAMES},
                reload_verified=bool(item["reload_verified"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{metrics_path} contains an invalid fold record.") from exc
        exported_folds.append(artifact)
        fold_report = _format_fold_report(artifact, total_folds)
        if write_run_log:
            _write_log(run_log, fold_report, echo=echo)
        if artifact.seed not in exported_seeds:
            _write_log(seed_log_dir / f"seed_{artifact.seed}.log", f"Results for seed={artifact.seed}", echo=False)
            exported_seeds.add(artifact.seed)
        _write_log(seed_log_dir / f"seed_{artifact.seed}.log", fold_report, echo=False)

    seed_summaries = _summarise_seed_results(exported_folds)
    for item in seed_summaries:
        seed_report = _format_summary_report(
            item["summary"],
            f"Seed {item['seed']} summary across {total_folds} folds (mean +/- std)",
        )
        _write_log(seed_log_dir / f"seed_{item['seed']}.log", seed_report, echo=False)
        if write_run_log:
            _write_log(run_log, seed_report, echo=echo)
    if write_run_log:
        _write_log(
            run_log,
            _format_summary_report(
                _summarise_seed_means(seed_summaries),
                f"Final summary across {len(seed_summaries)} seed means (mean +/- std)",
            ),
            echo=echo,
        )
    return directory


def evaluate_checkpoint(checkpoint: str | Path, dataset_name: str, data_dir: str | Path, device_name: str) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}.")
    device = resolve_device(device_name)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    dataset = load_dataset(dataset_name, data_dir)
    validate_model_dataset(payload["model_name"], dataset)
    if tuple(payload["feature_dims"]) != dataset.feature_counts:
        raise ValueError("Checkpoint feature dimensions do not match the selected dataset.")
    test_idx = np.asarray(payload["test_idx"], dtype=np.int64)
    train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
    graph_source = payload.get("graph_source", graph_source_for_model(payload["model_name"]))
    graph_data, _ = _scaled_graph_data(
        dataset, load_all_edge_indexes(dataset, graph_source), train_idx, device, payload["scalers"]
    )
    embeddings = _reload_model(checkpoint, graph_data, device)
    classifier_file = checkpoint.with_name(checkpoint.stem + "_classifier.pkl")
    if payload["model_name"] == "without-contrastive":
        kwargs = payload["model_kwargs"]
        model = PSGCLWithoutContrastive(*dataset.feature_counts, **kwargs).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        with torch.no_grad():
            probability = torch.softmax(model(*graph_data), dim=1).cpu().numpy()
        prediction = probability.argmax(axis=1)
    elif classifier_file.is_file():
        with classifier_file.open("rb") as handle:
            classifier = pickle.load(handle)
        prediction = classifier.predict(embeddings)
        probability = classifier.predict_proba(embeddings)
    else:
        raise FileNotFoundError(
            f"Expected classifier sidecar is missing: {classifier_file}. The checkpoint alone is not "
            "enough to reproduce the original embedding-classifier evaluation."
        )
    metric_values = classification_metrics(
        dataset.labels[test_idx], prediction[test_idx], probability[test_idx],
        embeddings[test_idx], dataset.n_classes,
    )
    return {
        "checkpoint": str(checkpoint),
        "model": payload["model_name"],
        "dataset": dataset.name,
        "test_samples": int(test_idx.size),
        "metrics": metric_values,
        "reload_verified": True,
    }
