from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def clean_gene_name(name: str) -> str:
    gene = str(name).strip()
    if gene.upper().startswith("ENSG"):
        return gene.split(".")[0].upper()
    return gene.replace(".", "-").upper()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_brca_inputs(
    gene_csv: Path,
    labels_path: Path,
    allow_positional_labels: bool,
) -> tuple[pd.DataFrame, Any, str]:
    gene_data = pd.read_csv(gene_csv, index_col=0).T
    if gene_data.empty:
        raise ValueError("The BRCA gene matrix is empty.")
    if gene_data.index.astype(str).duplicated().any():
        raise ValueError("The BRCA gene matrix has duplicate sample IDs.")

    labels_frame = pd.read_csv(labels_path)
    if labels_frame.empty:
        raise ValueError("The labels CSV is empty.")
    id_column = next(
        (
            column
            for column in labels_frame.columns
            if column.lower() in {"sample_id", "sampleid", "sample", "id"}
        ),
        None,
    )
    if id_column is not None:
        value_columns = [column for column in labels_frame.columns if column != id_column]
        if len(value_columns) != 1:
            raise ValueError("A labels CSV with sample IDs must contain exactly one label column.")
        labels_frame[id_column] = labels_frame[id_column].astype(str)
        if labels_frame[id_column].duplicated().any():
            raise ValueError("The labels CSV contains duplicate sample IDs.")
        gene_ids = gene_data.index.astype(str)
        if set(gene_ids) != set(labels_frame[id_column]):
            raise ValueError("Sample IDs in the gene matrix and labels CSV do not match exactly.")
        labels = labels_frame.set_index(id_column).loc[gene_ids, value_columns[0]].to_numpy()
        return gene_data, labels, "sample-id-verified"

    if not allow_positional_labels:
        raise ValueError(
            "The labels CSV has no sample-ID column, so alignment cannot be verified. "
            "Provide a sample_id column or pass --allow-positional-labels."
        )
    labels = labels_frame.iloc[:, 0].to_numpy()
    if len(gene_data) != len(labels):
        raise ValueError(
            f"Gene matrix has {len(gene_data)} samples but labels has {len(labels)} rows."
        )
    warnings.warn(
        "Using positional BRCA labels because the supplied CSV has no sample IDs.",
        stacklevel=2,
    )
    return gene_data, labels, "positional-unverified"


def select_brca_genes(
    gene_data: pd.DataFrame,
    labels: Any,
    output_dir: Path,
    top_n: int = 100,
    seed: int = 777,
    provenance: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if top_n < 1:
        raise ValueError("--top-n must be positive.")
    if len(gene_data) != len(labels):
        raise ValueError("gene_data and labels have different lengths.")
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; choose a new output directory.")
    output_dir.mkdir(parents=True)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
        oob_score=True,
    )
    rf.fit(gene_data, labels)

    importance = (
        pd.DataFrame(
            {"raw_gene": gene_data.columns.astype(str), "gini_importance": rf.feature_importances_}
        )
        .sort_values("gini_importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.insert(0, "rank", importance.index + 1)
    importance["clean_gene"] = importance["raw_gene"].map(clean_gene_name)
    importance["selected"] = importance["rank"] <= top_n
    importance.to_csv(output_dir / "gini_importance_full.csv", index=False)

    selected = importance.loc[importance["selected"]].copy()
    selected.to_csv(output_dir / "selected_genes.csv", index=False)
    invalid = selected.loc[selected["clean_gene"].eq("")].copy()
    invalid.assign(reason="empty_after_legacy_cleanup").to_csv(
        output_dir / "genes_rejected_before_enrichr.csv", index=False
    )

    manifest = {
        "analysis": "BRCA Random-Forest gene selection",
        "sample_count": int(gene_data.shape[0]),
        "feature_count": int(gene_data.shape[1]),
        "top_n": int(top_n),
        "random_forest": {
            "n_estimators": 500,
            "max_depth": None,
            "random_state": int(seed),
            "n_jobs": -1,
            "oob_score": True,
            "observed_oob_score": float(rf.oob_score_),
            "importance": "scikit-learn impurity-based (Gini) feature_importances_",
        },
        "identifier_handling": {
            "cleanup": "strip; uppercase; remove ENSG version; replace periods with hyphens",
            "external_symbol_mapping": "not performed",
            "unique_submission_is_deferred_to_R": True,
        },
        "provenance": provenance or {},
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the established BRCA Random Forest and export genes for R Enrichr."
    )
    parser.add_argument("--gene-csv", type=Path, required=True, help="Features × samples CSV; first column is gene ID.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--label-source",
        required=True,
        choices=("actual", "psgcl-predicted", "external"),
        help="Provenance declaration only; it does not alter labels.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--allow-positional-labels", action="store_true")
    return parser


def _main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.gene_csv.is_file() or not args.labels.is_file():
        raise FileNotFoundError("Both --gene-csv and --labels must name existing files.")
    gene_data, labels, alignment = load_brca_inputs(
        args.gene_csv, args.labels, args.allow_positional_labels
    )
    selected = select_brca_genes(
        gene_data,
        labels,
        args.output_dir,
        args.top_n,
        args.seed,
        {
            "gene_csv": str(args.gene_csv),
            "gene_csv_sha256": _sha256(args.gene_csv),
            "labels": str(args.labels),
            "labels_sha256": _sha256(args.labels),
            "label_source_declared_by_user": args.label_source,
            "alignment": alignment,
        },
    )
    print(
        f"Selected {len(selected)} ranked features with 500-tree Random Forest; "
        f"handoff: {args.output_dir / 'selected_genes.csv'}"
    )


def main(argv: list[str] | None = None) -> None:
    try:
        _main(argv)
    except (FileNotFoundError, FileExistsError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
