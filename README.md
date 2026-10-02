# P-SGCL

This repository provides the implementation of our proposed model, P-SGCL,
for cancer-subtype classification using gene expression, DNA methylation,
and miRNA data. It also includes baseline models and ablation experiments.

![P-SGCL architecture](images/architecture_19.png)

## Setup

The code has been verified with Python 3.11. Run the following commands in Bash
on Linux, macOS, or Windows with WSL:

```bash
git clone https://github.com/anhnqak57/P-SGCL.git
cd P-SGCL

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

Install PyTorch for CPU:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

**Optional — CUDA 12.4:** On Linux or WSL with a compatible NVIDIA GPU and
driver, use this command instead of the CPU installation:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

## Repository structure

```text
P-SGCL/
├── run.py                   # central CLI for P-SGCL, MCRGCN, and central ablations
├── build_graphs.py          # creates adaptive-percentile graphs
├── requirements.txt        # training and evaluation dependencies
├── psgcl_core/              # data loading, preprocessing, runner, metrics, and CLI
├── models/                  # P-SGCL and graph encoders
├── baselines/               # MCRGCN, MOGONET, and MCGNN
├── ablations/               # without-contrastive and psgcl-mcrgcn-graph
├── analysis/brca/           # optional BRCA gene selection and enrichment
└── dataset/                 # source MAT files and edge lists for each dataset
```

## Datasets

The supplied datasets are under `dataset/`. P-SGCL and MCRGCN support `brca`,
`brca-v5`, `gbm`, and `lgg`. MOGONET and MCGNN support `BRCA` (four labels),
`BRCA-v5` (five labels), `GBM`, and `LGG`.

Validate the inputs before training:

```bash
python run.py inspect-data --dataset lgg --data-dir dataset
```

## Proposed model: P-SGCL

Use a unique `--run-name` for each experiment. Existing run outputs are not
overwritten.

Train P-SGCL on LGG:

```bash
python run.py train --model psgcl --dataset lgg --run-name psgcl_lgg_full
```

## Baselines

### MCRGCN

```bash
python run.py train --model mcrgcn --dataset gbm --run-name mcrgcn_gbm_full
```

MCRGCN uses the supplied percentile graphs by default. To use the supplied
MCRGCN graphs on BRCA or GBM, set `--graph-source legacy-mcrgcn`:

```bash
python run.py train --model mcrgcn --dataset gbm \
  --graph-source legacy-mcrgcn --run-name mcrgcn_gbm_legacy
```

### MOGONET

```bash
python -m baselines.mogonet \
  --dataset LGG --data-dir dataset \
  --output-dir outputs/mogonet --run-name mogonet_lgg_full
```

### MCGNN

```bash
python -m baselines.mcgnn \
  --dataset LGG --data-dir dataset \
  --output-dir outputs/mcgnn --run-name mcgnn_lgg_full
```

For MOGONET and MCGNN, use `--dataset BRCA` for four-label BRCA or
`--dataset BRCA-v5` for five-label BRCA.

## Ablations

```bash
python -m ablations.without_contrastive \
  --config ablations/configs/without_contrastive_gbm.json \
  --run-name without_contrastive_gbm_full

python -m ablations.psgcl_mcrgcn_graph \
  --config ablations/configs/psgcl_mcrgcn_graph_gbm.json \
  --run-name psgcl_mcrgcn_graph_gbm_full
```

The `psgcl-mcrgcn-graph` ablation supports only four-class BRCA and GBM.

## Results and evaluation

Runs launched through `run.py` save results and checkpoints to
`outputs/<dataset>/<model>/<run-name>/`. Read `training.log` for summary results
and `seed_logs/` for per-seed fold results.

Reload a saved central fold:

```bash
python run.py evaluate \
  --dataset lgg --data-dir dataset \
  --checkpoint outputs/lgg/psgcl/psgcl_lgg_full/checkpoints/seed_223_fold_1.pt
```

Contrastive-model evaluation requires both the `.pt` checkpoint and its `.pkl`
classifier sidecar.

## Utility commands

Create adaptive-percentile graphs in a new output directory:

```bash
python build_graphs.py \
  --dataset lgg --data-dir dataset \
  --output-dir generated_graphs/lgg_adaptive_percentile
```

Show CLI options:

```bash
python run.py --help
```

Optional BRCA gene-selection and enrichment analyses are in `analysis/brca/`.
