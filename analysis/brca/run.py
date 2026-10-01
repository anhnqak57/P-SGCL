from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import pandas as pd

from .enrichment import run_brca_enrichment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank BRCA genes and run KEGG/GO enrichment.")
    parser.add_argument("--gene-csv", type=Path, required=True, help="Features x samples CSV with gene names in column 1.")
    parser.add_argument("--labels", type=Path, required=True, help="One-column label CSV aligned with the gene CSV samples.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--padj-threshold", type=float, default=0.05)
    parser.add_argument(
        "--allow-positional-labels", action="store_true",
        help="Explicitly allow a one-column labels CSV with no sample IDs."
    )
    return parser


def _main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.top_n < 1:
        raise ValueError("--top-n must be positive.")
    if not 0 < args.padj_threshold <= 1:
        raise ValueError("--padj-threshold must be in (0, 1].")
    if not args.gene_csv.is_file() or not args.labels.is_file():
        raise FileNotFoundError("Both --gene-csv and --labels must name existing files.")
    if args.output_dir.exists():
        raise FileExistsError("--output-dir already exists; choose a new directory to retain prior analysis outputs.")
    gene_data = pd.read_csv(args.gene_csv, index_col=0).T
    labels_frame = pd.read_csv(args.labels)
    id_column = next(
        (column for column in labels_frame.columns if column.lower() in {"sample_id", "sampleid", "sample", "id"}),
        None,
    )
    if id_column is not None:
        value_columns = [column for column in labels_frame.columns if column != id_column]
        if len(value_columns) != 1:
            raise ValueError("A labelled CSV with sample IDs must contain exactly one label column.")
        labels_frame[id_column] = labels_frame[id_column].astype(str)
        if labels_frame[id_column].duplicated().any():
            raise ValueError("The labels CSV contains duplicate sample IDs.")
        gene_ids = gene_data.index.astype(str)
        if set(gene_ids) != set(labels_frame[id_column]):
            raise ValueError("Sample IDs in the gene matrix and labels CSV do not match exactly.")
        labels = labels_frame.set_index(id_column).loc[gene_ids, value_columns[0]].to_numpy()
    else:
        if not args.allow_positional_labels:
            raise ValueError(
                "The labels CSV has no sample-ID column, so alignment cannot be verified. "
                "Provide a sample_id column or pass --allow-positional-labels to acknowledge the existing order."
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
    run_brca_enrichment(
        gene_data=gene_data,
        y_model_labels=labels,
        output_dir=str(args.output_dir),
        top_n=args.top_n,
        seed=args.seed,
        padj_threshold=args.padj_threshold,
    )


def main(argv: list[str] | None = None) -> None:
    try:
        _main(argv)
    except (FileNotFoundError, FileExistsError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
