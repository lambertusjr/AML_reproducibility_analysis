
# Reproducible Evaluation of Graph-Based and Tabular Models for AML Detection

Companion code for **"How Comparable Are Graph-Based AML Detection
Results? An Audit of Reporting Practice Against a Reproducibility
Standard"**, Lambertus van Zyl, *IEEE Access*, 2026.

This repository contains the full evaluation pipeline behind the paper:
dataset preprocessing, per-`(dataset, model)` hyperparameter
optimisation, and seeded final evaluation across **seven models** and
**six datasets**, with all metrics and precision–recall curves saved to
disk. The published optimal hyperparameters are shipped in
[`best_hyperparameters.json`](best_hyperparameters.json), so the final
evaluation can be reproduced **without** re-running the search.

-   **Paper:** [link / DOI — TODO]
-   **Datasets:** Elliptic, IBM-AML (HI/LI × Small/Medium), AMLSim — not
    redistributed here; see [Data](#data).

------------------------------------------------------------------------

## Overview

The task is **node classification** (illicit vs. licit) under a strict
**temporal split**. For each `(dataset, model)` pair the pipeline:

1.  Loads/preprocesses the dataset into a PyTorch Geometric graph with
    temporal masks.
2.  *(optional)* Runs Optuna (TPE, 200 trials) to maximise **validation
    PR-AUC**.
3.  Refits on `train ∪ val` and evaluates on the held-out `test` split
    across **10 seeds**.
4.  Saves per-run metrics and PR-curve artefacts.

The study is framed around reproducibility: results characterise
seed-level variance and ranking stability rather than declaring a
winning model.

------------------------------------------------------------------------

## Repository structure

| File | Role |
|------------------------------------|------------------------------------|
| `main.py` | Orchestrator: preprocess → (optional) tune → seeded final evaluation. |
| `config.py` | Single source of truth for dataset paths, model groups, loader-vs-full-batch sets, and global constants. |
| `pre_processing.py` | PyG `InMemoryDataset` classes: `EllipticDataset`, unified `IBMAMLDataset` (+ thin per-variant subclasses), `AMLSimDataset`. |
| `pre_process_datasets.py` | One-shot helper to materialise every processed dataset. |
| `models.py` | `ModelWrapper` (train/eval paths) + `GCN`, `GAT` (GATv2), `GIN`, `MLP`. |
| `helper_functions.py` | Search spaces, metrics, memory-aware batch-size probe, study bookkeeping. |
| `funcs_for_optuna.py` | Optuna study setup, pruning, and the objective. |
| `training_functions.py` | Training loops (full-batch / `NeighborLoader` / in-VRAM) with PR-AUC checkpointing. |
| `testing.py` | Seeded final evaluation; loads published configs from `best_hyperparameters.json`; metric + PR-curve serialisation. |
| `utilities.py` | `FocalLoss`, seeding, memory monitoring, batch-size cache. |
| `dependencies.py` | Shared imports. |
| `best_hyperparameters.json` | Published optimal hyperparameters for all 42 `(dataset, model)` combinations. |
| `environment.yml` | Conda environment specification. |

------------------------------------------------------------------------

## Models and datasets

**Models (7):** `MLP`, `GCN`, `GAT` (GATv2), `GIN` (PyTorch Geometric);
`SVM` (linear `SGDClassifier`, hinge loss), `RF` (scikit-learn); `XGB`
(XGBoost).

**Datasets (6):** `Elliptic`, `IBM_AML_HiSmall`, `IBM_AML_LiSmall`,
`IBM_AML_HiMedium`, `IBM_AML_LiMedium`, `AMLSim`.

| Dataset | Loading strategy | Split |
|-------------------|------------------------------------|------------------|
| Elliptic | GPU full-batch (`perf_eval_mask` for unlabelled nodes) | temporal, timesteps 1–30 / 31–40 / 41–49 |
| IBM-AML (all four) | CPU `NeighborLoader` (GNNs), fanout `[10, 5]` | 60/20/20 chronological |
| AMLSim | CPU `NeighborLoader` (GNNs) | 60/20/20, by account first-seen time |

The MLP uses an **in-VRAM** path on `NeighborLoader` datasets (it
ignores the graph, so no sampling is needed).

------------------------------------------------------------------------

## Installation

Requires an NVIDIA GPU. Reference environment: **Python 3.10, PyTorch
2.4.1, CUDA 11.8** (the versions the published results were produced
with).

``` bash
# 1. Base environment (conda / mamba)
conda env create -f environment.yml
conda activate aml-repro

# 2. PyTorch Geometric + CUDA extensions (matching wheel index for torch 2.4 / cu118)
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

# 3. (optional) pyg-lib — fast C++ NeighborLoader sampler (2–5× over torch-sparse).
#    Safe to skip; the code falls back to the torch-sparse sampler automatically.
pip install pyg_lib -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
```

> Secondary packages (numpy, pandas, scikit-learn, xgboost, optuna, …)
> are not version-pinned in `environment.yml`; conda-forge resolves them
> against the pinned torch/python. For an exact lock, run
> `conda list --export` in the built environment and commit the result.

------------------------------------------------------------------------

## Data {#data}

The datasets are **not** included (size and licensing). Download each
from the sources below and place the raw files under the PyG `raw/`
convention. The first run preprocesses them into `processed/data.pt`.

| Dataset | Source |
|--------------------------------------|----------------------------------|
| Elliptic | <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set> |
| IBM-AML | <https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml> |
| AMLSim | <https://www.kaggle.com/datasets/anshankul/ibm-amlsim-example-dataset> |

Expected layout (note the `_raw` filename suffix the loaders require):

```         
Datasets/
├── Elliptic_dataset/raw/
│   ├── elliptic_txs_features_raw.csv
│   ├── elliptic_txs_classes_raw.csv
│   └── elliptic_txs_edgelist_raw.csv
├── IBM_AML_dataset/
│   ├── HiSmall/raw/HI-Small_Trans_raw.csv
│   ├── LiSmall/raw/LI-Small_Trans_raw.csv
│   ├── HiMedium/raw/HI-Medium_Trans_raw.csv
│   └── LiMedium/raw/LI-Medium_Trans_raw.csv
└── AMLSim_dataset/raw/
    ├── accounts.csv
    ├── transactions.csv
    └── alerts.csv
```

The loaders accept both the Kaggle-native filenames (e.g.
`elliptic_txs_features.csv`, `HI-Small_Trans.csv`) and the legacy
`*_raw.csv` names shown above, so no rename is required — drop the files
in as downloaded. The directory tree (one `raw/` folder per
dataset/variant) is the only requirement.

Materialise everything once:

``` bash
python pre_process_datasets.py
```

Then verify your processed graphs match the published reference:

``` bash
python verify_graphs.py --check
```

A `PASS` for every dataset confirms your preprocessing is byte-identical
to the reference (`hash_reference.json`). A `FAIL` points to the
specific tensor that diverged — see the script's diagnostic output for
likely causes (raw-file version, dependency versions, or the HI-Small
dtype map).

------------------------------------------------------------------------

## Running

**Reproduce the final evaluation from the published configs** (no
re-tuning):

``` bash
python main.py IBM_AML_HiSmall GCN          # one (dataset, model)
python main.py                              # the defaults set in main.py
```

`testing.py` reads the model's hyperparameters from
`best_hyperparameters.json`, refits on `train ∪ val`, and evaluates over
10 seeds.

**Re-run the hyperparameter search from scratch** (optional; see note
below):

``` bash
python main.py IBM_AML_HiSmall GCN --retune
```

**On a cluster:** the repository contains no scheduler scripts. Submit
jobs with your own scheduler, passing the dataset and model as
positional arguments (`python main.py DATASET MODEL`), and size
resources to the dataset (Medium variants use `NeighborLoader` and
benefit from extra system RAM).

> Re-tuning is not guaranteed to reproduce the exact published configs:
> Optuna explores a large stochastic search space. The
> deterministic-up-to-seed claim applies to the **evaluation from the
> published configs**, not to the search.

------------------------------------------------------------------------

## Hyperparameter search space

Optimiser: **Optuna**, TPE sampler, `direction="maximize"`, **200
trials** per `(dataset, model)`, SQLite-backed studies. Objective:
**validation PR-AUC** (`average_precision_score`). Pruning: Hyperband
(`min_resource=20`, `max_resource=500`, `reduction_factor=3`) for the
neural models; none for the tree/linear models.

**Shared — neural models** (`MLP`, `GCN`, `GAT`, `GIN`):

| Parameter                    | Range       | Scale  |
|------------------------------|-------------|--------|
| `learning_rate`              | 1e-4 – 5e-2 | log    |
| `weight_decay`               | 1e-5 – 1e-2 | log    |
| `gamma_focal` (Focal loss γ) | 0.1 – 5.0   | linear |
| `n_epochs`                   | 5 – 500     | linear |

**Per-model:**

| Model | Parameter | Range / Choices |
|------------------|----------------------|--------------------------------|
| MLP | `hidden_units` | 32 – 256 |
|  | `dropout_1`, `dropout_2` | 0.0 – 0.7 |
| GCN | `hidden_units` | 32 – 256 |
|  | `dropout` | 0.0 – 0.7 |
|  | `n_layers` | 1 – 3 |
| GAT | `hidden_units` | 32 – 256 |
|  | `num_heads` | 1 – 8 |
|  | `dropout_1`, `dropout_2`, `feature_dropout` | 0.0 – 0.7 |
|  | `n_layers` | 1 – 3 |
| GIN | `hidden_units` | 32 – 256 |
|  | `dropout` | 0.0 – 0.5 |
|  | `n_layers` | 1 – 3 |
| SVM (`SGDClassifier`, hinge, L2) | `C` | 0.01 – 100.0 (log); `alpha = 1/(C·N)`, `class_weight="balanced"` |
| RF | `n_estimators` | 50 – 500 (step 50) |
|  | `max_depth` | 5 – 15 |
|  | `min_samples_split` | 2 – 20 |
|  | `min_samples_leaf` | 1 – 10 |
|  | `max_features` | {`sqrt`, `log2`}; `class_weight="balanced"` |
| XGB | `max_depth` | 5 – 15 |
|  | `Gamma_XGB` | 0 – 5 |
|  | `n_estimators` | 50 – 500 (step 50) |
|  | `learning_rate_XGB` | 0.001 – 0.3 (log) |
|  | `colsample_bytree`, `subsample` | 0.5 – 1.0 |
|  | `min_child_weight` | 1 – 10 |
|  | `reg_alpha`, `reg_lambda` | 1e-8 – 10.0 (log) |
|  | fixed | `tree_method="hist"`, `scale_pos_weight = neg/pos` (train only) |

The selected values for every `(dataset, model)` combination are in
[`best_hyperparameters.json`](best_hyperparameters.json).

------------------------------------------------------------------------

## Evaluation protocol

-   **Seeds:** 10 independent retrainings per `(model, dataset)`.
-   **Final fit:** trained on `train ∪ val`; final-epoch weights
    evaluated on `test`.
-   **Loss:** Focal loss with inverse-frequency `alpha` and tuned
    `gamma`.
-   **Metrics:** F1-illicit, PR-AUC, ROC-AUC, precision/recall (weighted
    and illicit-class), accuracy, MCC, and Cohen's κ.
-   **Primary metric:** PR-AUC (also the optimisation objective).

### Outputs

```         
results/<dataset>/
├── pr_curves/   # <model>_run_<i>_<job>_pr_{data.npz, curve.png}
├── metrics/
└── pkl_files/   # per-run metrics, detailed & summary DataFrames
```

------------------------------------------------------------------------

## Hardware

Developed on an RTX 4080 SUPER / Ryzen 9 9950X3D workstation (Windows +
WSL2) and a Linux GPU cluster. The batch-size probe and memory monitors
keep the `NeighborLoader` datasets from paging to disk on
memory-constrained machines.

------------------------------------------------------------------------

## Acknowledgments

Parts of the dataset-loading pipeline were initially adapted from the
MIT-licensed code released by Deprez et al. (2025). The 4-hour
temporal-window edge construction for the IBM-AML datasets follows the
method introduced by:

H. Tariq and M. Hassani, "Topology-agnostic detection of temporal money
laundering flows in billion-scale transactions," in *Joint European
Conference on Machine Learning and Knowledge Discovery in Databases*,
Springer, 2023, pp. 402–419.

## License

See [LICENSE](LICENSE).

## Citation

``` bibtex
@article{vanzyl2026howcomparable,
  title   = {How Comparable Are Graph-Based AML Detection Results? An Audit of Reporting Practice Against a Reproducibility Standard},
  author  = {van Zyl, Lambertus},
  journal = {IEEE Access},
  year    = {2026}
}
```
