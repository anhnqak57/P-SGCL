from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from utils import PERCENTILE_LIST, build_percentile_graph

from .data import load_dataset


def build_graphs(dataset_name: str, data_dir: str | Path, output_dir: str | Path) -> Path:
    dataset = load_dataset(dataset_name, data_dir)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(
            f"{output} already exists. Choose a new --output-dir; supplied graphs are never overwritten."
        )
    output.mkdir(parents=True)
    manifest: dict[str, object] = {
        "dataset": dataset.name,
        "source_mat": str(dataset.source_mat),
        "algorithm": "utils.build_percentile_graph",
        "candidate_percentiles": [float(value) for value in PERCENTILE_LIST],
        "views": {},
    }
    for name, view in zip(("gene", "methy", "mirna"), dataset.views):
        adjacency, chosen_percentile, threshold, matrix_path = build_percentile_graph(
            view,
            type=f"{dataset.name}_{name}",
            f_name=output,
        )
        edge_path = output / f"edges_{name}_{dataset.name}.csv"
        rows, columns = np.nonzero(adjacency)
        np.savetxt(edge_path, np.column_stack([rows, columns]), fmt="%d", delimiter=",")
        manifest["views"][name] = {
            "shape": list(view.shape),
            "chosen_percentile": chosen_percentile,
            "threshold": threshold,
            "matrix": matrix_path.name,
            "edges": edge_path.name,
            "edge_count": int(rows.size),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output
