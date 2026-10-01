# Ablation experiments

Two ablations are implemented and supported by the central cross-validation runner. Both use the same fold-local scaling, metrics, checkpoints, and reload verification as P-SGCL.

| Variant | What changes | Supported datasets | Graph source |
| --- | --- | --- | --- |
| without-contrastive | Replaces P-SGCL contrastive loss and external embedding classifier with a 60 → 30 → C supervised in-model classifier. The three encoders and projectors remain. | brca, brca-v5, gbm, lgg | percentile |
| psgcl-mcrgcn-graph | Keeps the complete P-SGCL architecture, contrastive objective, equal fusion, and external MLP. Only the three supplied edge lists change. | brca and gbm with exactly four source-MAT classes | legacy-mcrgcn |

## P-SGCL with MCRGCN graphs

This is a graph-source ablation, not MCRGCN. The runner selects dataset/<dataset>/graphs/legacy-mcrgcn/ directly and records graph_source plus every resolved edge path in config.json and each checkpoint. Evaluation reads that stored graph source, rather than defaulting to percentile graphs.

The supplied edge-row counts are:

| Dataset | Gene | Methylation | miRNA |
| --- | ---: | ---: | ---: |
| brca | 1,016 | 4,122 | 11,868 |
| gbm | 1,400 | 582 | 1,756 |

The command rejects brca-v5 and lgg before training. It also rejects BRCA or GBM if their source labels no longer resolve to four classes.

~~~bash
python -m ablations.psgcl_mcrgcn_graph \
  --config ablations/configs/psgcl_mcrgcn_graph_brca.json \
  --run-name psgcl_mcrgcn_graph_brca_full

python -m ablations.psgcl_mcrgcn_graph \
  --config ablations/configs/psgcl_mcrgcn_graph_gbm.json \
  --run-name psgcl_mcrgcn_graph_gbm_full
~~~

## Without contrastive learning

~~~bash
python -m ablations.without_contrastive \
  --config ablations/configs/without_contrastive_gbm.json \
  --run-name without_contrastive_gbm_full
~~~

Use the same dataset, folds, seeds, dimensions, classifier settings, and output-independent evaluation protocol when comparing an ablation with P-SGCL. Generated output directories are intentionally ignored by Git.
