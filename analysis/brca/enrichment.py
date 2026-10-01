from __future__ import annotations

import re
import textwrap
import warnings
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


SEED = 777
TOP_N = 100
PADJ_THRESHOLD = 0.05

GENE_SETS = [
    "KEGG_2021_Human",
    "GO_Biological_Process_2023",
    "GO_Cellular_Component_2023",
    "GO_Molecular_Function_2023",
]

LIBRARY_CONFIG = {
    "KEGG_2021_Human": {
        "short_name": "kegg",
        "title": "KEGG Pathway",
        "color": "steelblue",
        "bar_top_n": 15,
        "dot_top_n": 10,
    },
    "GO_Biological_Process_2023": {
        "short_name": "go_bp",
        "title": "GO Biological Process",
        "color": "steelblue",
        "bar_top_n": 10,
        "dot_top_n": 10,
    },
    "GO_Cellular_Component_2023": {
        "short_name": "go_cc",
        "title": "GO Cellular Component",
        "color": "seagreen",
        "bar_top_n": 10,
        "dot_top_n": 10,
    },
    "GO_Molecular_Function_2023": {
        "short_name": "go_mf",
        "title": "GO Molecular Function",
        "color": "darkorange",
        "bar_top_n": 10,
        "dot_top_n": 10,
    },
}

BRCA_KEYWORDS = [
    "breast cancer",
    "estrogen",
    "erbb",
    "her2",
    "pi3k-akt",
    "ras signaling",
    "mapk",
    "mtor",
    "p53",
    "cell cycle",
    "apoptosis",
    "foxo",
    "homologous recombination",
    "dna repair",
]


plt.rcParams.update(
    {
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "font.weight": "normal",
    }
)


def clean_gene_name(name: str) -> str:
    gene = str(name).strip()

    if gene.upper().startswith("ENSG"):


        return gene.split(".")[0].upper()


    return gene.replace(".", "-").upper()


def unique_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def select_significant_terms(
    result_df: pd.DataFrame,
    library_name: str,
    top_n: int,
    padj_threshold: float,
) -> pd.DataFrame:
    selected = result_df[result_df["Gene_set"] == library_name].copy()
    selected["Adjusted P-value"] = pd.to_numeric(
        selected["Adjusted P-value"], errors="coerce"
    )
    selected = selected.dropna(subset=["Term", "Adjusted P-value"])
    selected = selected[selected["Adjusted P-value"] < padj_threshold]
    selected = selected.nsmallest(top_n, "Adjusted P-value").copy()

    if selected.empty:
        return selected

    selected["-log10(padj)"] = -np.log10(
        selected["Adjusted P-value"].clip(lower=np.finfo(float).tiny)
    )
    selected["Display_term"] = (
        selected["Term"]
        .str.replace(r"\s*\(GO:\d+\)\s*$", "", regex=True)
        .apply(lambda term: textwrap.fill(str(term), width=45))
    )

    if "Overlap" in selected.columns:
        overlap = selected["Overlap"].astype(str).str.extract(
            r"(?P<Count>\d+)\s*/\s*(?P<Term_size>\d+)"
        )
        selected["Count"] = pd.to_numeric(overlap["Count"], errors="coerce")
        selected["Term_size"] = pd.to_numeric(
            overlap["Term_size"], errors="coerce"
        )
        selected["Overlap_ratio"] = selected["Count"] / selected["Term_size"]

    return selected


def plot_pvalue_bar(
    term_df: pd.DataFrame,
    title: str,
    color: str,
    output_path: Path,
    padj_threshold: float,
) -> None:
    if term_df.empty:
        print(f"No significant terms found for {title}.")
        return

    plot_df = term_df.sort_values("-log10(padj)", ascending=True)
    fig_height = max(6.0, 0.55 * len(plot_df) + 2.0)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.barh(
        plot_df["Display_term"],
        plot_df["-log10(padj)"],
        color=color,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.88,
    )
    ax.axvline(
        -np.log10(padj_threshold),
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"FDR = {padj_threshold:g}",
    )
    ax.set_xlabel(r"$-\log_{10}(\mathrm{adjusted}\ P\mathrm{-value})$")
    ax.set_title(f"{title} Enrichment in TCGA-BRCA")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_dotplot(
    term_df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    required = {"Overlap_ratio", "Count", "-log10(padj)"}
    if term_df.empty:
        print(f"No significant terms found for {title}.")
        return
    if not required.issubset(term_df.columns):
        print(f"Overlap information is unavailable for {title}.")
        return

    plot_df = term_df.dropna(subset=list(required)).copy()
    plot_df = plot_df.sort_values("Overlap_ratio", ascending=True)
    if plot_df.empty:
        print(f"No valid overlap values found for {title}.")
        return

    fig_height = max(6.5, 0.62 * len(plot_df) + 2.0)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    scatter = ax.scatter(
        x=plot_df["Overlap_ratio"],
        y=plot_df["Display_term"],
        s=plot_df["Count"] * 40,
        c=plot_df["-log10(padj)"],
        cmap="Reds",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.85,
    )

    ax.set_xlabel("Overlap ratio (selected genes / term size)")
    ax.set_title(f"{title} Enrichment in TCGA-BRCA")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r"$-\log_{10}(\mathrm{adjusted}\ P\mathrm{-value})$")

    count_values = sorted(plot_df["Count"].astype(int).unique())
    legend_counts = sorted(
        {
            count_values[0],
            count_values[len(count_values) // 2],
            count_values[-1],
        }
    )
    legend_handles = [
        ax.scatter(
            [],
            [],
            s=count * 40,
            color="gray",
            edgecolors="black",
            alpha=0.85,
            label=str(count),
        )
        for count in legend_counts
    ]
    ax.legend(
        handles=legend_handles,
        title="Gene Count",
        loc="lower right",
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def build_gene_term_table(
    library_df: pd.DataFrame,
    term_column_name: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for _, row in library_df.dropna(subset=["Genes"]).iterrows():
        for gene in str(row["Genes"]).split(";"):
            records.append(
                {
                    "Gene": gene.strip().upper(),
                    term_column_name: row["Term"],
                    "Gene_set": row["Gene_set"],
                    "Adjusted_P": row["Adjusted P-value"],
                    "Overlap": row.get("Overlap", np.nan),
                }
            )

    return pd.DataFrame(records)


def run_brca_enrichment(
    gene_data: pd.DataFrame,
    y_model_labels,
    output_dir: str = "./outputs/brca_python_enrichr",
    top_n: int = TOP_N,
    seed: int = SEED,
    padj_threshold: float = PADJ_THRESHOLD,
) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not isinstance(gene_data, pd.DataFrame):
        raise TypeError("gene_data must be a pandas DataFrame.")
    if gene_data.empty:
        raise ValueError("gene_data is empty.")
    if len(gene_data) != len(y_model_labels):
        raise ValueError("gene_data and y_model_labels have different lengths.")

    X = gene_data
    y = y_model_labels

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
        oob_score=True,
    )
    rf.fit(X, y)

    feature_importance_df = (
        pd.DataFrame(
            {
                "gene": X.columns,
                "gini_importance": rf.feature_importances_,
            }
        )
        .sort_values("gini_importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance_df.to_csv(
        output_path / "gini_importance_full_v2.csv", index=False
    )

    top_genes_raw = feature_importance_df["gene"].head(top_n).tolist()
    if any(str(gene).upper().startswith("ENSG") for gene in top_genes_raw):
        warnings.warn(
            "Ensembl identifiers were detected. Map them to HGNC gene symbols "
            "before Enrichr analysis for reliable annotation.",
            stacklevel=2,
        )

    top_genes_clean = unique_in_order(
        [clean_gene_name(gene) for gene in top_genes_raw]
    )
    pd.DataFrame(
        {
            "raw_gene": top_genes_raw,
            "clean_gene": [clean_gene_name(gene) for gene in top_genes_raw],
        }
    ).to_csv(output_path / f"top_{top_n}_genes_v2.csv", index=False)

    enr = gp.enrichr(
        gene_list=top_genes_clean,
        gene_sets=GENE_SETS,
        organism="human",
        outdir=None,
        cutoff=padj_threshold,
    )
    result_df = (
        enr.results.sort_values("Adjusted P-value").reset_index(drop=True)
    )
    result_df.to_csv(output_path / "enrichment_kegg_go_full_v2.csv", index=False)

    matched_genes = set()
    for genes_string in result_df["Genes"].dropna():
        matched_genes.update(
            gene.strip().upper() for gene in str(genes_string).split(";")
        )
    top_gene_set = set(top_genes_clean)
    mapped_genes = matched_genes & top_gene_set
    match_rate = len(mapped_genes) / len(top_gene_set) if top_gene_set else 0.0

    print(f"OOB score: {rf.oob_score_:.4f}")
    print(f"Selected unique genes: {len(top_gene_set)}")
    print(f"Genes matched in KEGG/GO results: {len(mapped_genes)}")
    print(f"Overall match rate: {match_rate:.2%}")

    kegg_df = result_df[result_df["Gene_set"] == "KEGG_2021_Human"].copy()
    kegg_df.to_csv(output_path / "kegg_enrichment_full_v2.csv", index=False)

    keyword_pattern = "|".join(re.escape(keyword) for keyword in BRCA_KEYWORDS)
    brca_relevant = kegg_df[
        kegg_df["Term"]
        .fillna("")
        .str.lower()
        .str.contains(keyword_pattern, regex=True)
    ].copy()
    brca_relevant.to_csv(
        output_path / "brca_relevant_kegg_pathways_v2.csv", index=False
    )

    kegg_gene_table = build_gene_term_table(kegg_df, "KEGG_Pathway")
    kegg_gene_table.to_csv(
        output_path / "gene_to_kegg_pathway_v2.csv", index=False
    )

    go_df = result_df[
        result_df["Gene_set"].fillna("").str.startswith("GO_")
    ].copy()
    go_gene_table = build_gene_term_table(go_df, "GO_Term")
    go_gene_table.to_csv(output_path / "gene_to_go_term_v2.csv", index=False)

    plotted_terms: dict[str, pd.DataFrame] = {}
    for library_name, config in LIBRARY_CONFIG.items():
        library_full = result_df[result_df["Gene_set"] == library_name].copy()
        library_full.to_csv(
            output_path / f"{config['short_name']}_enrichment_full_v2.csv",
            index=False,
        )

        bar_df = select_significant_terms(
            result_df=result_df,
            library_name=library_name,
            top_n=int(config["bar_top_n"]),
            padj_threshold=padj_threshold,
        )
        dot_df = select_significant_terms(
            result_df=result_df,
            library_name=library_name,
            top_n=int(config["dot_top_n"]),
            padj_threshold=padj_threshold,
        )
        plotted_terms[library_name] = bar_df
        bar_df.to_csv(
            output_path / f"{config['short_name']}_significant_terms_v2.csv",
            index=False,
        )

        plot_pvalue_bar(
            term_df=bar_df,
            title=str(config["title"]),
            color=str(config["color"]),
            output_path=output_path / f"{config['short_name']}_pvalue_barplot.png",
            padj_threshold=padj_threshold,
        )
        plot_dotplot(
            term_df=dot_df,
            title=str(config["title"]),
            output_path=output_path / f"{config['short_name']}_dotplot_19_final.png",
        )

    return {
        "random_forest": rf,
        "feature_importance": feature_importance_df,
        "top_genes": top_genes_clean,
        "enrichment_results": result_df,
        "plotted_terms": plotted_terms,
        "match_rate": match_rate,
    }









