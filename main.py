import os
# Must be set BEFORE any torch/CUDA import to take effect.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import gc
import json
import argparse

from dependencies import *
from utilities import *
from helper_functions import check_study_existence, ensure_edge_index_column_sorted
from funcs_for_optuna import hyperparameter_tuning
from testing import run_final_evaluation
import optuna
import config

configure_gpu_memory_limits(fraction=0.75)


# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="AML reproducibility pipeline: evaluate (default) or re-tune.",
)
parser.add_argument("dataset", nargs="?", default=None,
                    help=f"Dataset name (one of {list(config.DATASET_PATHS)}). "
                         f"Omit to use the defaults set in this file.")
parser.add_argument("model", nargs="?", default=None,
                    help="Model name (MLP, GCN, GAT, GIN, SVM, RF, XGB). "
                         "Omit to use the defaults set in this file.")
parser.add_argument("--retune", action="store_true",
                    help="Run the Optuna hyperparameter search before evaluation. "
                         "Default: skip search and evaluate using the published "
                         f"configs in {config.BEST_HYPERPARAMETERS_FILE}.")
parser.add_argument("--seed", type=int, default=None,
                    help="Fixed global seed. Default: a fresh random seed per run.")
args = parser.parse_args()

# Seeding (per-run reproducibility is handled inside testing.py; this seeds the
# orchestration itself, e.g. the batch-size probe).
if args.seed is not None:
    set_seed(args.seed)
    print(f"Seeded run with seed {args.seed}")
else:
    seed = np.random.SeedSequence().generate_state(1)[0]
    set_seed(seed)

# Datasets / models to process.
if args.dataset and args.model:
    datasets = [args.dataset]
    models = [args.model]
else:
    # Defaults — edit these for a local batch run.
    datasets = ["IBM_AML_HiSmall"]
    models = ["MLP"]
    # Full set:
    # datasets = ["Elliptic", "IBM_AML_HiSmall", "IBM_AML_LiSmall",
    #             "IBM_AML_HiMedium", "IBM_AML_LiMedium", "AMLSim"]
    # models   = ["MLP", "GCN", "GAT", "GIN", "SVM", "RF", "XGB"]

dataset_paths = config.DATASET_PATHS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_dataset(name):
    """Instantiate the processed PyG dataset for a given dataset name."""
    match name:
        case "Elliptic":
            from pre_processing import EllipticDataset
            return EllipticDataset(root=dataset_paths["Elliptic"])[0]
        case "IBM_AML_HiSmall":
            from pre_processing import IBMAMLDataset_HiSmall
            return IBMAMLDataset_HiSmall(root=dataset_paths["IBM_AML_HiSmall"])[0]
        case "IBM_AML_LiSmall":
            from pre_processing import IBMAMLDataset_LiSmall
            return IBMAMLDataset_LiSmall(root=dataset_paths["IBM_AML_LiSmall"])[0]
        case "IBM_AML_HiMedium":
            from pre_processing import IBMAMLDataset_HiMedium
            return IBMAMLDataset_HiMedium(root=dataset_paths["IBM_AML_HiMedium"])[0]
        case "IBM_AML_LiMedium":
            from pre_processing import IBMAMLDataset_LiMedium
            return IBMAMLDataset_LiMedium(root=dataset_paths["IBM_AML_LiMedium"])[0]
        case "AMLSim":
            from pre_processing import AMLSimDataset
            return AMLSimDataset(root=dataset_paths["AMLSim"])[0]
        case _:
            raise ValueError(f"Unknown dataset '{name}'. "
                             f"Choose from {list(dataset_paths)}.")


def prepare(data, name):
    """Sort edges, strip masks, and place on the right device for the strategy."""
    data = ensure_edge_index_column_sorted(data, name=name)
    data, masks = extract_and_remove_masks(data)
    # Full-batch datasets go to GPU; NeighborLoader datasets stay on CPU so the
    # full graph doesn't consume VRAM.
    if name not in config.LOADER_DATASETS:
        data = data.to(device)
        print(f"Data moved to device: {device}")
    else:
        print(f"Data kept on CPU for NeighborLoader-based processing ({name})")
    return data, masks


def cleanup():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


print(f"Processing {len(datasets)} dataset(s): {', '.join(datasets)}")
print(f"Mode: {'RE-TUNE then evaluate' if args.retune else 'EVALUATE from published configs'}")
print("=" * 80)

# ── Optional hyperparameter tuning ───────────────────────────────────────────
if args.retune:
    for idx, dataset in enumerate(datasets, 1):
        print(f"\n{'='*80}\nHyperparameter tuning {idx}/{len(datasets)}: {dataset}\n{'='*80}\n")

        if all(check_study_existence(model, dataset) for model in models):
            print(f"All studies for {dataset} are complete. Skipping.")
            continue

        data = load_dataset(dataset)
        print(f"Dataset {dataset} loaded for tuning.")
        data, masks = prepare(data, dataset)

        hyperparameter_tuning(models=models, dataset_name=dataset, data=data, masks=masks)

        del data, masks
        cleanup()
        print(f"\nCompleted tuning {idx}/{len(datasets)}: {dataset}")

    print(f"\n{'='*80}\nHYPERPARAMETER TUNING COMPLETE\n{'='*80}")

# ── Final evaluation ─────────────────────────────────────────────────────────
# Load best params either from the just-run Optuna studies (--retune) or, by
# default, from the published JSON so a clone reproduces the evaluation without
# the .db study files.
best_hp_json = None
if not args.retune:
    with open(config.BEST_HYPERPARAMETERS_FILE) as f:
        best_hp_json = json.load(f)

for idx, dataset in enumerate(datasets, 1):
    print(f"\n{'='*80}\nFinal evaluation {idx}/{len(datasets)}: {dataset}\n{'='*80}\n")

    model_parameters = {}
    for model_name in models:
        if args.retune:
            study_name = f'{model_name}_optimization on {dataset} dataset'
            storage_url = f'sqlite:///optimization_results_on_{dataset}_{model_name}.db'
            try:
                study = optuna.load_study(study_name=study_name, storage=storage_url)
                model_parameters[model_name] = [study.best_trial.params]
                print(f"Loaded tuned params for {model_name} "
                      f"(best val PR-AUC: {study.best_value:.4f})")
            except KeyError:
                print(f"No completed study for {model_name} on {dataset}, skipping.")
        else:
            if dataset in best_hp_json and model_name in best_hp_json[dataset]:
                model_parameters[model_name] = [best_hp_json[dataset][model_name]]
                print(f"Loaded published params for {model_name} on {dataset}.")
            else:
                print(f"No published params for {model_name} on {dataset} in "
                      f"{config.BEST_HYPERPARAMETERS_FILE}, skipping.")

    if not model_parameters:
        print(f"No parameters found for {dataset}. Skipping evaluation.")
        continue

    data = load_dataset(dataset)
    data, masks = prepare(data, dataset)

    run_final_evaluation(
        models=list(model_parameters.keys()),
        model_parameters=model_parameters,
        data=data,
        data_for_optimisation=dataset,
        masks=masks,
    )

    del data, masks, model_parameters
    cleanup()
    print(f"\nCompleted evaluation {idx}/{len(datasets)}: {dataset}")

print(f"\n{'='*80}\nALL PROCESSING COMPLETE\n{'='*80}")
