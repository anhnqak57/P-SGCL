# Dataset layout

The tracked dataset directory contains the inputs required by current runners.

~~~text
dataset/<dataset>/
├── source/                       # required source MAT
├── graphs/percentile/            # required edge CSVs for standard P-SGCL/MCRGCN
├── graphs/legacy-mcrgcn/         # BRCA/GBM only; used by psgcl-mcrgcn-graph
└── features/                     # BRCA-v5 RF/Enrichr inputs; LGG legacy CSVs are excluded
~~~

The runner reads source MATs and edge CSVs. It does not read graph matrix MAT files, so those generated/redundant artifacts are not part of the Git-ready dataset.

Use the default from the repository root:

~~~bash
python run.py inspect-data --dataset lgg --data-dir dataset
~~~

The loader accepts --data-dir . as well when the current directory contains dataset/. Do not change source-MAT sample order without regenerating the matching edge lists.

For BRCA-v5, the graph edge suffix is brca because the MAT prefix is BRCA; it is not brca-v5.

