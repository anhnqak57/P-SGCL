from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio
import torch


@dataclass(frozen=True)
class DatasetSpec:
    """Description of one dataset layout included in this repository."""

    name: str
    mat_file: str
    mat_prefix: str


DATASETS: dict[str, DatasetSpec] = {
    "brca": DatasetSpec("brca", "BRCA.mat", "BRCA"),
    "brca-v5": DatasetSpec("brca-v5", "BRCA_v5_labels.mat", "BRCA"),
    "gbm": DatasetSpec("gbm", "GBM.mat", "GBM"),
    "lgg": DatasetSpec("lgg", "LGG.mat", "LGG"),
}

GRAPH_SOURCES = ("percentile", "legacy-mcrgcn")


@dataclass
class OmicsDataset:
    """Validated aligned three-omics data in samples-by-features orientation."""

    name: str
    source_mat: Path
    gene: np.ndarray
    methylation: np.ndarray
    mirna: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    raw_label_values: np.ndarray
    edge_paths: tuple[Path, Path, Path]

    @property
    def n_samples(self) -> int:
        return int(self.labels.shape[0])

    @property
    def n_classes(self) -> int:
        return int(np.unique(self.labels).size)

    @property
    def feature_counts(self) -> tuple[int, int, int]:
        return tuple(int(x.shape[1]) for x in self.views)

    @property
    def views(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.gene, self.methylation, self.mirna


def supported_dataset_names() -> tuple[str, ...]:
    return tuple(DATASETS)


def _dataset_spec(dataset: str) -> DatasetSpec:
    key = dataset.lower().replace("_", "-")
    if key not in DATASETS:
        choices = ", ".join(supported_dataset_names())
        raise ValueError(f"Unsupported dataset {dataset!r}. Available datasets: {choices}.")
    return DATASETS[key]


def dataset_directory(dataset: str, data_dir: str | Path = "dataset") -> Path:
    spec = _dataset_spec(dataset)
    root = Path(data_dir)
    direct = root / spec.name
    nested = root / "dataset" / spec.name
    if direct.is_dir() or not nested.is_dir():
        return direct
    return nested


def dataset_mat_path(dataset: str, data_dir: str | Path = "dataset") -> Path:
    spec = _dataset_spec(dataset)
    mat_path = dataset_directory(spec.name, data_dir) / "source" / spec.mat_file
    if not mat_path.is_file():
        raise FileNotFoundError(
            f"Dataset {spec.name!r} requires {mat_path}. Use --data-dir dataset "
            "(or the project root containing dataset/) with the documented layout."
        )
    return mat_path


def resolve_dataset(dataset: str, data_dir: str | Path = "dataset") -> DatasetSpec:
    spec = _dataset_spec(dataset)
    dataset_mat_path(spec.name, data_dir)
    return spec


def _mat_value(mat: dict, key: str, source: Path) -> np.ndarray:
    if key not in mat:
        available = ", ".join(k for k in mat if not k.startswith("__"))
        raise ValueError(
            f"{source} has no variable {key!r}. Available variables: {available}."
        )
    return np.asarray(mat[key])


def _normalise_labels(raw_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw_labels).reshape(-1)
    if values.size == 0:
        raise ValueError("The label vector is empty.")
    raw_values, encoded = np.unique(values, return_inverse=True)
    if raw_values.size < 2:
        raise ValueError("At least two classes are required for stratified evaluation.")
    return encoded.astype(np.int64), raw_values


def load_dataset(dataset: str, data_dir: str | Path = "dataset") -> OmicsDataset:
    root = Path(data_dir).resolve()
    spec = resolve_dataset(dataset, root)
    source = dataset_mat_path(spec.name, root)
    mat = sio.loadmat(source)
    prefix = spec.mat_prefix

    gene = _mat_value(mat, f"{prefix}_Gene_Expression", source).T
    methylation = _mat_value(mat, f"{prefix}_Methy_Expression", source).T
    mirna = _mat_value(mat, f"{prefix}_Mirna_Expression", source).T
    labels, raw_label_values = _normalise_labels(
        _mat_value(mat, f"{prefix}_clinicalMatrix", source)
    )
    sample_ids = _mat_value(mat, f"{prefix}_indexes", source).reshape(-1)

    views = (gene, methylation, mirna)
    if any(view.ndim != 2 for view in views):
        raise ValueError("Each omics matrix must be two-dimensional.")
    sample_sizes = {view.shape[0] for view in views} | {labels.size, sample_ids.size}
    if len(sample_sizes) != 1:
        details = ", ".join(
            [f"gene={gene.shape[0]}", f"methylation={methylation.shape[0]}",
             f"mirna={mirna.shape[0]}", f"labels={labels.size}",
             f"sample_ids={sample_ids.size}"]
        )
        raise ValueError(f"Sample alignment failed for {source}: {details}.")
    if not all(np.isfinite(view).all() for view in views):
        raise ValueError(f"{source} contains NaN or infinity in an omics matrix.")
    if np.unique(sample_ids).size != sample_ids.size:
        raise ValueError(f"{source} contains duplicate sample identifiers.")

    edge_paths = _edge_paths_for(spec, dataset_directory(spec.name, root), "percentile")
    _require_edge_paths(edge_paths, "percentile")

    return OmicsDataset(
        name=spec.name,
        source_mat=source,
        gene=np.asarray(gene, dtype=np.float32),
        methylation=np.asarray(methylation, dtype=np.float32),
        mirna=np.asarray(mirna, dtype=np.float32),
        labels=labels,
        sample_ids=sample_ids,
        raw_label_values=raw_label_values,
        edge_paths=edge_paths,
    )


def _edge_paths_for(
    spec: DatasetSpec,
    directory: Path,
    graph_source: str,
) -> tuple[Path, Path, Path]:
    if graph_source not in GRAPH_SOURCES:
        choices = ", ".join(GRAPH_SOURCES)
        raise ValueError(f"Unsupported graph source {graph_source!r}. Available sources: {choices}.")
    edge_root = directory / "graphs" / graph_source
    suffix = spec.mat_prefix.lower()
    return (
        edge_root / f"edges_gene_{suffix}.csv",
        edge_root / f"edges_methy_{suffix}.csv",
        edge_root / f"edges_mirna_{suffix}.csv",
    )


def _require_edge_paths(edge_paths: tuple[Path, Path, Path], graph_source: str) -> None:
    missing = [str(path) for path in edge_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"The {graph_source!r} sample graphs are missing: " + "; ".join(missing) +
            ". Run build_graphs.py only when deliberately regenerating percentile graphs."
        )


def graph_edge_paths(dataset: OmicsDataset, graph_source: str = "percentile") -> tuple[Path, Path, Path]:
    if graph_source == "percentile":
        return dataset.edge_paths
    spec = _dataset_spec(dataset.name)
    paths = _edge_paths_for(spec, dataset.source_mat.parent.parent, graph_source)
    _require_edge_paths(paths, graph_source)
    return paths


def load_edge_index(path: str | Path, n_samples: int) -> torch.Tensor:
    path = Path(path)
    try:
        raw = np.loadtxt(path, delimiter=",", dtype=np.int64)
    except ValueError as exc:
        raise ValueError(f"Cannot parse edge list {path}; expected two integers per row.") from exc
    if raw.size == 0:
        raise ValueError(f"Edge list {path} is empty.")
    raw = np.atleast_2d(raw)
    if raw.shape[1] != 2:
        raise ValueError(f"Edge list {path} must have exactly two columns; got {raw.shape[1]}.")
    if raw.min() < 0 or raw.max() >= n_samples:
        raise ValueError(
            f"Edge list {path} has node indices outside [0, {n_samples - 1}]."
        )
    return torch.as_tensor(raw.T.copy(), dtype=torch.long).contiguous()


def load_all_edge_indexes(
    dataset: OmicsDataset,
    graph_source: str = "percentile",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    paths = graph_edge_paths(dataset, graph_source)
    return tuple(load_edge_index(path, dataset.n_samples) for path in paths)


def validate_folds(labels: np.ndarray, folds: int) -> None:
    if folds < 2:
        raise ValueError("--folds must be at least 2.")
    _, counts = np.unique(labels, return_counts=True)
    smallest = int(counts.min())
    if folds > smallest:
        raise ValueError(
            f"--folds={folds} is invalid: the smallest class has only {smallest} samples."
        )


def parse_seeds(seed: int | None, seeds: Iterable[int] | None) -> tuple[int, ...]:
    if seeds is not None:
        parsed = tuple(int(value) for value in seeds)
        if not parsed:
            raise ValueError("--seeds must contain at least one integer.")
        return parsed
    return (777 if seed is None else int(seed),)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except RuntimeError as exc:
        raise ValueError(f"Invalid --device value {requested!r}.") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu or --device auto.")
    return device
