"""
Central configuration shared across the pipeline: dataset paths, model groups,
data-loading strategy, and global constants. Importing these from one place
removes the previously duplicated ``loader_datasets`` / model-group sets that
lived in main.py, funcs_for_optuna.py, and testing.py.
"""

# --- Datasets -------------------------------------------------------------
DATASET_PATHS = {
    "Elliptic":         "Datasets/Elliptic_dataset",
    "IBM_AML_HiSmall":  "Datasets/IBM_AML_dataset/HiSmall",
    "IBM_AML_LiSmall":  "Datasets/IBM_AML_dataset/LiSmall",
    "IBM_AML_HiMedium": "Datasets/IBM_AML_dataset/HiMedium",
    "IBM_AML_LiMedium": "Datasets/IBM_AML_dataset/LiMedium",
    "AMLSim":           "Datasets/AMLSim_dataset",
}
DATASET_NAMES = list(DATASET_PATHS)

# Datasets trained with NeighborLoader mini-batching. Everything except
# Elliptic, which is trained full-batch on the GPU.
LOADER_DATASETS = {
    "AMLSim", "IBM_AML_HiSmall", "IBM_AML_LiSmall",
    "IBM_AML_HiMedium", "IBM_AML_LiMedium",
}
FULL_BATCH_DATASETS = {"Elliptic"}

# --- Models ---------------------------------------------------------------
WRAPPER_MODELS = {"MLP", "GCN", "GAT", "GIN"}   # PyTorch / PyG models
SKLEARN_MODELS = {"SVM", "RF", "XGB"}           # scikit-learn / XGBoost models

# --- Search / evaluation --------------------------------------------------
N_TRIALS = 200                   # Optuna trials per (dataset, model)
N_SEEDS = 10                     # final-evaluation retrainings
MAX_N_EPOCHS = 500               # upper bound on the tuned n_epochs
MLP_IN_VRAM_BATCH_SIZE = 16384   # MLP in-VRAM path batch size
NUM_NEIGHBORS = [10, 5]          # NeighborLoader fanout

BEST_HYPERPARAMETERS_FILE = "best_hyperparameters.json"
