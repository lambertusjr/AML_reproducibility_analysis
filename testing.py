import os
import secrets
import torch
import numpy as np
import pandas as pd
import optuna
import gc
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from helper_functions import (
    _get_model_instance, balanced_class_weights, find_optimal_batch_size,
    calculate_metrics, calculate_pr_metrics_batched, save_pr_artifacts,
    save_metrics_to_pickle, save_dataframe_to_pickle, neighbor_loader_kwargs
)
from training_functions import (train_and_validate, train_and_validate_with_loader,
                                 train_and_validate_in_vram)
from utilities import FocalLoss, set_seed, load_batch_size_by_phase
from models import ModelWrapper
from torch_geometric.loader import NeighborLoader
from config import (WRAPPER_MODELS, SKLEARN_MODELS, LOADER_DATASETS, N_SEEDS,
                    MLP_IN_VRAM_BATCH_SIZE, NUM_NEIGHBORS)

# Output-filename suffix. Only appended when a JOB_ID env var is set (e.g. on a
# cluster, to keep parallel jobs from clobbering each other). On a workstation it
# is empty, giving clean names like GCN_run_1_metrics.pkl.
JOB_ID = os.environ.get("JOB_ID")
_SUFFIX = f"_{JOB_ID}" if JOB_ID else ""


def run_final_evaluation(models, model_parameters, data, data_for_optimisation, masks):
    """
    Entry point: iterates over models and dispatches to evaluate_model_performance.
    """
    print(f"\nStarting FINAL EVALUATION for {data_for_optimisation} dataset...")

    # Create output directories once
    os.makedirs(f"results/{data_for_optimisation}/pr_curves", exist_ok=True)
    os.makedirs(f"results/{data_for_optimisation}/metrics", exist_ok=True)
    os.makedirs(f"results/{data_for_optimisation}/pkl_files", exist_ok=True)

    for model_name in models:
        if model_name not in model_parameters or not model_parameters[model_name]:
            print(f"No parameters found for {model_name}, skipping evaluation.")
            continue

        best_params = model_parameters[model_name][-1]
        print(f"Evaluating {model_name} on {data_for_optimisation}...")

        evaluate_model_performance(
            model_name=model_name,
            best_params=best_params,
            data=data,
            masks=masks,
            dataset_name=data_for_optimisation,
            n_runs=10
        )


def evaluate_model_performance(model_name, best_params, data, masks, dataset_name, n_runs=N_SEEDS):
    """
    Runs n_runs iterations of a single model+dataset combination, collects
    detailed per-run metrics, and saves summary statistics.

    Handles three model families:
        - wrapper_models  (MLP, GCN, GAT, GIN)  → PyTorch training loop
        - sklearn_models  (SVM, RF)              → scikit-learn .fit / .predict
        - gpu_sklearn_models (XGB)               → XGBoost on GPU tensors

    And three data-loading strategies:
        - NeighborLoader   (IBM_AML_HiMedium / LiMedium)
        - Full-batch       (AMLSim, IBM_AML_HiSmall / LiSmall)
        - Full-batch Elliptic (special mask handling)
    """

    # ── 0. Constants & device ────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wrapper_models  = WRAPPER_MODELS
    sklearn_models  = SKLEARN_MODELS
    gpu_sklearn_models = set()          # retained for the dead XGB-on-GPU branch
    batch_loader_datasets = LOADER_DATASETS

    # ── 1. Validate masks vs data ────────────────────────────────────────
    # masks is a dict from extract_and_remove_masks:
    #   {'train_mask', 'val_mask', 'test_mask',
    #    'train_perf_eval_mask', 'val_perf_eval_mask', 'test_perf_eval_mask'}
    regular_masks = [masks["train_mask"], masks["val_mask"], masks["test_mask"]]
    train_mask, val_mask, test_mask = regular_masks

    is_elliptic = (dataset_name == "Elliptic")

    # ── 2. Prepare sklearn data (lazy, only if needed) ───────────────────
    sklearn_data = None
    if model_name in sklearn_models or model_name in gpu_sklearn_models:
        needs_cpu = model_name in sklearn_models  # SVM/RF need numpy
        convert = lambda t: t.cpu().numpy() if needs_cpu else t

        # The final fit trains on train ∪ val, matching the wrapper models
        # (which merge train and val via combined_train_mask before the final
        # fit). The 'train' split below therefore uses the combined mask so
        # sklearn models see the same amount of training data.
        combined_train_mask = train_mask | val_mask
        split_masks = {'train': combined_train_mask, 'val': val_mask, 'test': test_mask}
        sklearn_data = {}
        for split, split_mask in split_masks.items():
            sklearn_data[f'{split}_x'] = convert(data.x[split_mask])
            sklearn_data[f'{split}_y'] = convert(data.y[split_mask])
        if is_elliptic:
            combined_train_perf_mask = (masks['train_perf_eval_mask']
                                        | masks['val_perf_eval_mask'])
            split_perf_masks = {'train': combined_train_perf_mask,
                                'val': masks['val_perf_eval_mask'],
                                'test': masks['test_perf_eval_mask']}
            for split, split_perf_mask in split_perf_masks.items():
                sklearn_data[f'{split}_perf_x'] = convert(data.x[split_perf_mask])
                sklearn_data[f'{split}_perf_y'] = convert(data.y[split_perf_mask])

    # ── 3. Class weights for focal loss ──────────────────────────────────
    if sklearn_data is not None:
        # sklearn_data['train_y'] may be numpy (SVM/RF); balanced_class_weights expects a tensor
        alpha_focal = balanced_class_weights(torch.as_tensor(sklearn_data['train_y'], dtype=torch.long))
    else:
        alpha_focal = balanced_class_weights(data.y[train_mask])
    alpha_focal = alpha_focal.to(device)

    # ── 4. Determine loading strategy ────────────────────────────────────
    # MLP on loader datasets uses the in-VRAM path: masked feature/label
    # tensors are pre-moved to GPU and iterated by index slicing, with no
    # NeighborLoader involvement at all.
    is_mlp_in_vram = (model_name == "MLP" and dataset_name in batch_loader_datasets)
    use_loader = (dataset_name in batch_loader_datasets) and (model_name in wrapper_models) \
                 and not is_mlp_in_vram
    train_loader, val_loader, test_loader = None, None, None
    x_train_in_vram = y_train_in_vram = x_test_in_vram = y_test_in_vram = None

    if is_mlp_in_vram:
        # Final-fit trains on train ∪ val (val intentionally None inside the
        # training loop). data.x lives on CPU for loader datasets, so index on
        # CPU and move only the selected slice to GPU.
        combined_train_mask = train_mask | val_mask
        x_train_in_vram = data.x[combined_train_mask].to(device, non_blocking=True)
        y_train_in_vram = data.y[combined_train_mask].to(device, non_blocking=True)
        x_test_in_vram = data.x[test_mask].to(device, non_blocking=True)
        y_test_in_vram = data.y[test_mask].to(device, non_blocking=True)
        del combined_train_mask
        print(f"  > Using IN-VRAM path for {model_name} on {dataset_name} "
              f"(batch_size={MLP_IN_VRAM_BATCH_SIZE}, no NeighborLoader)")

    if use_loader:
        num_neighbors = NUM_NEIGHBORS

        def _model_builder():
            class MockTrial:
                def suggest_int(self, name, low, high, step=None): return high
                def suggest_float(self, name, low, high, log=False): return low
                def suggest_categorical(self, name, choices): return choices[0]
            return _get_model_instance(MockTrial(), model_name, data, device,
                                       train_mask=train_mask)

        # --- Evaluation batch size ---
        print(f"Finding optimal EVALUATION batch size for {model_name}...")
        eval_batch_size = load_batch_size_by_phase(dataset_name, model_name, phase='evaluation')
        if eval_batch_size is None:
            eval_batch_size = find_optimal_batch_size(
                _model_builder, data, device, train_mask,
                num_neighbors=num_neighbors, dataset_name=dataset_name,
                model_name=model_name, phase='evaluation'
            )
        print(f"  > Using EVALUATION batch size {eval_batch_size} for {model_name}")

        # --- Training batch size ---
        tuning_batch_size = load_batch_size_by_phase(dataset_name, model_name, phase='tuning')
        if tuning_batch_size is None:
            print(f"Finding optimal TUNING batch size for {model_name}...")
            tuning_batch_size = find_optimal_batch_size(
                _model_builder, data, device, train_mask,
                num_neighbors=num_neighbors, dataset_name=dataset_name,
                model_name=model_name, phase='tuning'
            )
        print(f"  > Using TUNING batch size {tuning_batch_size} for {model_name}")

        # Final-fit trains on train ∪ val; val_loader is intentionally None so the
        # training loop skips per-epoch validation (matches B-Deprez's protocol).
        combined_train_mask = train_mask | val_mask
        loader_kwargs = neighbor_loader_kwargs()
        train_loader = NeighborLoader(data, num_neighbors=num_neighbors,
                                      batch_size=tuning_batch_size,
                                      input_nodes=combined_train_mask,
                                      is_sorted=True, **loader_kwargs)
        val_loader   = None
        test_loader  = NeighborLoader(data, num_neighbors=num_neighbors,
                                      batch_size=eval_batch_size,
                                      input_nodes=test_mask,
                                      is_sorted=True, **loader_kwargs)
    else:
        print(f"  > Using FULL BATCH for {model_name} on {dataset_name} (no NeighborLoader)")

    # ── 5. Prepare masks dict for train_and_validate (full-batch) ────────
    #    For the final-test fit we train on train ∪ val and pass val_mask=None
    #    so the training loop skips per-epoch checkpointing (B-Deprez style).
    if not use_loader and model_name in wrapper_models and not is_mlp_in_vram:
        if is_elliptic:
            combined_train_mask = masks['train_mask'] | masks['val_mask']
            combined_train_perf_mask = masks['train_perf_eval_mask'] | masks['val_perf_eval_mask']
            training_masks_dict = {
                'train_mask': combined_train_mask,
                'train_perf_eval_mask': combined_train_perf_mask,
                'val_mask': None,
                'val_perf_eval_mask': None,
            }
        else:
            training_masks_dict = {
                'train_mask': masks['train_mask'] | masks['val_mask'],
                'val_mask': None,
            }

    # ── 6. Seeds ─────────────────────────────────────────────────────────
    # All models use the same run count (N_SEEDS) for a uniform variance estimate.
    seeds = [secrets.randbits(32) for _ in range(n_runs)]

    # ══════════════════════════════════════════════════════════════════════
    # ── 7. MAIN RUN LOOP ─────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    detailed_metrics = []

    for i, seed in enumerate(seeds):
        set_seed(seed)
        print(f"  > Run {i+1}/{n_runs} (Seed {seed})")

        fixed_trial = optuna.trial.FixedTrial(best_params)
        model_instance = _get_model_instance(fixed_trial, model_name, data, device,
                                             train_mask=train_mask)

        # sklearn / XGB models carry their own RNG and do not pick up the
        # per-run seed from set_seed (SVM even hardcodes random_state=42). Seed
        # each instance with this run's seed so every one of the n_runs uses a
        # distinct seed, matching the wrapper models.
        if model_name in sklearn_models or model_name in gpu_sklearn_models:
            model_instance.random_state = seed

        y_pred, y_probs = None, None
        run_metrics = {}

        # ── 7a. Wrapper models (MLP / GCN / GAT / GIN) ──────────────────
        if model_name in wrapper_models:
            learning_rate = best_params.get('learning_rate', 0.01)
            weight_decay  = best_params.get('weight_decay', 5e-4)
            gamma_focal   = best_params.get('gamma_focal', 2.0)

            criterion = FocalLoss(alpha=alpha_focal, gamma=gamma_focal, reduction='mean')
            optimiser = torch.optim.Adam(model_instance.parameters(),
                                         lr=learning_rate, weight_decay=weight_decay)

            model_wrapper = ModelWrapper(model_instance, optimiser, criterion)
            model_wrapper.model.to(device)

            # Use the tuned n_epochs (5–500). Fallback for any pre-migration study.
            num_epochs = best_params.get('n_epochs', 200 if model_name == "MLP" else 150)

            # ── TRAIN (no per-epoch validation: val_loader/val_mask is None) ─
            if is_mlp_in_vram:
                best_wts, _ = train_and_validate_in_vram(
                    model_wrapper, x_train_in_vram, y_train_in_vram,
                    None, None, num_epochs, MLP_IN_VRAM_BATCH_SIZE
                )
            elif use_loader:
                best_wts, _ = train_and_validate_with_loader(
                    model_wrapper, train_loader, val_loader, num_epochs
                )
            else:
                best_wts, _ = train_and_validate(
                    model_wrapper, data, training_masks_dict,
                    num_epochs, dataset_name
                )

            # Load trained weights (final-epoch weights since val is None)
            model_wrapper.model.load_state_dict(best_wts)

            # ── EVALUATE ─────────────────────────────────────────────────
            if is_mlp_in_vram:
                _, test_metrics = model_wrapper.evaluate_in_vram(
                    x_test_in_vram, y_test_in_vram, MLP_IN_VRAM_BATCH_SIZE
                )

            elif use_loader:
                _, test_metrics = model_wrapper.evaluate_loader(test_loader)

            elif is_elliptic:
                # evaluate_elliptic expects [test_mask, test_perf_eval_mask]
                elliptic_test_masks = [masks['test_mask'],
                                       masks['test_perf_eval_mask']]
                _, test_metrics = model_wrapper.evaluate_elliptic(data, elliptic_test_masks)

            else:
                # evaluate_full expects a single mask
                _, test_metrics = model_wrapper.evaluate_full(data, test_mask)

            y_probs    = test_metrics['probs']
            y_pred     = test_metrics['preds']
            y_true     = test_metrics['y_true']
            run_metrics = test_metrics

        # ── 7b. Sklearn models (SVM / RF) ────────────────────────────────
        elif model_name in sklearn_models:
            if is_elliptic:
                train_x_key, train_y_key = 'train_perf_x', 'train_perf_y'
                test_x_key,  test_y_key  = 'test_perf_x',  'test_perf_y'
            else:
                train_x_key, train_y_key = 'train_x', 'train_y'
                test_x_key,  test_y_key  = 'test_x',  'test_y'

            model_instance.fit(sklearn_data[train_x_key], sklearn_data[train_y_key])
            y_true = sklearn_data[test_y_key]

            try:
                y_probs = model_instance.predict_proba(sklearn_data[test_x_key])
            except AttributeError:
                # Fallback for SVM: scale decision_function to [0, 1]
                if hasattr(model_instance, 'decision_function'):
                    dfunc = model_instance.decision_function(sklearn_data[test_x_key])
                    y_probs = (dfunc - dfunc.min()) / (dfunc.max() - dfunc.min())
                else:
                    y_probs = None

            y_pred = model_instance.predict(sklearn_data[test_x_key])

            if y_probs is not None:
                run_metrics = calculate_metrics(y_true, y_pred, y_probs)
            else:
                run_metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': np.nan, 'precision_illicit': np.nan,
                    'recall': np.nan, 'recall_illicit': np.nan,
                    'f1': f1_score(y_true, y_pred, average='weighted'),
                    'f1_illicit': f1_score(y_true, y_pred, pos_label=1, average='binary'),
                    'roc_auc': np.nan, 'PRAUC': np.nan,
                    'MCC': matthews_corrcoef(y_true, y_pred), 'kappa': np.nan,
                }

        # ── 7c. GPU sklearn models (XGB) ────────────────────────────────
        elif model_name in gpu_sklearn_models:
            if is_elliptic:
                train_x_key, train_y_key = 'train_perf_x', 'train_perf_y'
                test_x_key,  test_y_key  = 'test_perf_x',  'test_perf_y'
            else:
                train_x_key, train_y_key = 'train_x', 'train_y'
                test_x_key,  test_y_key  = 'test_x',  'test_y'

            model_instance.fit(sklearn_data[train_x_key], sklearn_data[train_y_key])
            y_true = sklearn_data[test_y_key]

            y_probs = model_instance.predict_proba(sklearn_data[test_x_key])
            y_pred  = model_instance.predict(sklearn_data[test_x_key])
            run_metrics = calculate_metrics(
                y_true.cpu().numpy() if hasattr(y_true, 'cpu') else y_true,
                y_pred.cpu().numpy() if hasattr(y_pred, 'cpu') else y_pred,
                y_probs.cpu().numpy() if hasattr(y_probs, 'cpu') else y_probs
            )

        # ── 8. PR curve computation (shared across all models) ───────────
        if y_probs is not None:
            # Ensure numpy arrays for consistency
            y_probs_np = y_probs.cpu().numpy() if hasattr(y_probs, 'cpu') else (
                y_probs if isinstance(y_probs, np.ndarray) else np.array(y_probs))
            y_true_np = y_true.cpu().numpy() if hasattr(y_true, 'cpu') else (
                y_true if isinstance(y_true, np.ndarray) else np.array(y_true))

            # Extract positive-class probabilities
            y_probs_for_pr = y_probs_np[:, 1] if y_probs_np.ndim == 2 else y_probs_np

            y_probs_tensor = torch.as_tensor(y_probs_for_pr, dtype=torch.float32)
            y_true_tensor  = torch.as_tensor(y_true_np, dtype=torch.long)

            precision, recall, thresholds = calculate_pr_metrics_batched(
                y_probs_tensor, y_true_tensor)

            pr_filename = f"results/{dataset_name}/pr_curves/{model_name}_run_{i+1}{_SUFFIX}"
            pr_auc = save_pr_artifacts(precision, recall, thresholds, pr_filename)
            run_metrics['PRAUC'] = pr_auc

            pr_data = {
                'precision':  precision.numpy(),
                'recall':     recall.numpy(),
                'thresholds': thresholds.numpy(),
                'auc':        pr_auc,
                'y_true':     y_true_np,
                'y_probs':    y_probs_for_pr,
            }
            save_metrics_to_pickle(
                pr_data,
                f"results/{dataset_name}/pkl_files/{model_name}_run_{i+1}_pr_data{_SUFFIX}.pkl"
            )

        # ── 9. Tag and store run metrics ─────────────────────────────────
        run_metrics['model'] = model_name
        run_metrics['run']   = i + 1

        save_metrics_to_pickle(
            run_metrics,
            f"results/{dataset_name}/pkl_files/{model_name}_run_{i+1}_metrics{_SUFFIX}.pkl"
        )
        # CREATE A LIGHTWEIGHT COPY for the DataFrame
        # Remove heavy tensors (probs, preds, y_true) before appending to the list
        lightweight_metrics = run_metrics.copy()
        keys_to_remove = ['probs', 'preds', 'y_true']
        for key in keys_to_remove:
            if key in lightweight_metrics:
                del lightweight_metrics[key]
        
        detailed_metrics.append(lightweight_metrics)

        # ── 10. Per-run cleanup ──────────────────────────────────────────
        del model_instance, y_pred, y_probs, run_metrics
        if model_name in wrapper_models:
            del model_wrapper, best_wts, optimiser, criterion
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        gc.collect()

    # Free MLP in-VRAM tensors once all runs are complete (kept across runs
    # since the masked feature data is identical for every seed).
    if is_mlp_in_vram:
        del x_train_in_vram, y_train_in_vram, x_test_in_vram, y_test_in_vram
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    # ── 11. AGGREGATE & SAVE ─────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    df = pd.DataFrame(detailed_metrics)
    save_dataframe_to_pickle(df, f"results/{dataset_name}/pkl_files/{model_name}_detailed_metrics{_SUFFIX}.pkl")

    columns_to_drop = {'model', 'run', 'probs', 'preds', 'y_true'}
    numeric_df = df.drop(columns=[col for col in columns_to_drop if col in df.columns],
                         errors='ignore')
    summary = numeric_df.agg(['mean', 'std']).transpose()
    summary['model'] = model_name

    save_dataframe_to_pickle(summary, f"results/{dataset_name}/pkl_files/{model_name}_summary_metrics{_SUFFIX}.pkl")

    print(f"  > Completed {model_name}. Pickled metrics saved to results/{dataset_name}/pkl_files/"
          + (f" (JOB_ID={JOB_ID})" if JOB_ID else ""))
