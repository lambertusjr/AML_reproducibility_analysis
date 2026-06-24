import optuna
import numpy as np
import torch
from tqdm import tqdm
from models import ModelWrapper
from helper_functions import (_get_model_instance, balanced_class_weights, check_study_existence,
                              find_optimal_batch_size, run_trial_with_cleanup,
                              neighbor_loader_kwargs)
from utilities import FocalLoss
from training_functions import (train_and_validate_with_loader, train_and_validate,
                                 train_and_validate_in_vram)
import pandas as pd
import os
from datetime import datetime
from torch_geometric.loader import NeighborLoader

from config import (MAX_N_EPOCHS, MLP_IN_VRAM_BATCH_SIZE, LOADER_DATASETS, FULL_BATCH_DATASETS,
                    WRAPPER_MODELS, SKLEARN_MODELS, N_TRIALS, NUM_NEIGHBORS)


def hyperparameter_tuning(models, dataset_name, data, masks):
    print(f"Starting hyperparameter tuning for dataset: {dataset_name}")
    model_parameters = {model_name: [] for model_name in models}

    for model_name in tqdm(models, desc="Models", unit="model"):
        study_name = f'{model_name}_optimization on {dataset_name} dataset'
        db_path = f'sqlite:///optimization_results_on_{dataset_name}_{model_name}.db'
        if check_study_existence(model_name, dataset_name):
            print(f"Study for {model_name} on {dataset_name} already complete. Skipping optimization.")
            continue

        # Find optimal batch size for NeighborLoader datasets with wrapper models.
        # MLP uses the in-VRAM path (no NeighborLoader, no graph sampling), so the
        # batch-size probe is unnecessary — we use a fixed MLP_IN_VRAM_BATCH_SIZE.
        batch_size = None
        if (dataset_name in LOADER_DATASETS and model_name in WRAPPER_MODELS
                and model_name != "MLP"):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            def _model_builder():
                class MockTrial:
                    def suggest_int(self, name, low, high, step=None): return high
                    def suggest_float(self, name, low, high, log=False): return low
                    def suggest_categorical(self, name, choices): return choices[0]
                return _get_model_instance(MockTrial(), model_name, data, device)

            batch_size = find_optimal_batch_size(
                _model_builder, data, device, masks['train_mask'],
                num_neighbors=NUM_NEIGHBORS, dataset_name=dataset_name,
                model_name=model_name, phase='tuning'
            )

        # Hyperband pruner kills trials that lag at intermediate epochs. Only
        # applied to wrapper models — sklearn trials don't report per-epoch
        # values, so the pruner is a no-op for them.
        pruner = (
            optuna.pruners.HyperbandPruner(
                min_resource=20, max_resource=MAX_N_EPOCHS, reduction_factor=3
            )
            if model_name in WRAPPER_MODELS
            else optuna.pruners.NopPruner()
        )
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=db_path,
            load_if_exists=True,
            pruner=pruner,
        )

        # Count every "settled" trial — COMPLETE + PRUNED + FAIL — so the
        # remaining count matches Optuna's n_trials semantics (one objective call
        # per trial, regardless of outcome).
        settled_states = {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        }
        num_completed = len([t for t in study.trials if t.state in settled_states])
        remaining_trials = max(0, N_TRIALS - num_completed)

        if remaining_trials == 0:
            print(f"Study for {model_name} on {dataset_name} already has "
                  f"{num_completed} completed trials. Skipping.")
            model_parameters[model_name].append(study.best_params)
            continue

        if num_completed > 0:
            print(f"Resuming optimization for {model_name} on {dataset_name}: "
                  f"{num_completed} trials done, {remaining_trials} remaining "
                  f"(target: {N_TRIALS}).")
        else:
            print(f"Starting optimization for {model_name} on {dataset_name} "
                  f"with {N_TRIALS} trials.")

        alpha_focal = balanced_class_weights(data.y[masks['train_mask']])
        with tqdm(total=N_TRIALS, initial=num_completed,
                  desc=f"{model_name} trials", leave=False, unit="trial") as trial_bar:
            def _optuna_progress_callback(study_inner, trial):
                trial_bar.update()

            study.optimize(
                lambda trial: run_trial_with_cleanup(
                    objective, model_name, trial, model_name, data,
                    alpha_focal=alpha_focal, dataset_name=dataset_name, masks=masks,
                    batch_size=batch_size, num_neighbors=NUM_NEIGHBORS),
                n_trials=remaining_trials,
                catch=(torch.OutOfMemoryError,),
                callbacks=[_optuna_progress_callback]
            )
        model_parameters[model_name].append(study.best_params)
        print(f"Best val PR-AUC for {model_name} on {dataset_name}: {study.best_value:.4f}")

    return model_parameters


def objective(trial, model, data, alpha_focal, dataset_name, masks, batch_size=None, num_neighbors=NUM_NEIGHBORS):
    train_mask = masks['train_mask']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Pass train_mask to _get_model_instance for XGB leakage prevention.
    model_instance = _get_model_instance(trial, model, data, device, train_mask=train_mask)

    if model in WRAPPER_MODELS:
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        gamma_focal = trial.suggest_float('gamma_focal', 0.1, 5.0)
        n_epochs = trial.suggest_int('n_epochs', 5, MAX_N_EPOCHS)

        if alpha_focal is None:
            alpha_focal = balanced_class_weights(data.y[train_mask]).to(device)

        if alpha_focal.shape[0] != 2:
            raise ValueError(f"alpha_focal should have 2 elements for binary classification, "
                             f"got {alpha_focal.shape[0]}")

        criterion = FocalLoss(alpha=alpha_focal, gamma=gamma_focal, reduction='mean')

        if dataset_name == "Elliptic":
            train_val_masks = {
                'train_mask': masks['train_mask'],
                'train_perf_eval_mask': masks['train_perf_eval_mask'],
                'val_mask': masks['val_mask'],
                'val_perf_eval_mask': masks['val_perf_eval_mask'],
            }
        else:
            train_val_masks = {
                'train_mask': masks['train_mask'],
                'val_mask': masks['val_mask'],
            }

        optimiser = torch.optim.Adam(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model_wrapper = ModelWrapper(model_instance, optimiser, criterion)
        model_wrapper.model.to(device)

        if dataset_name in LOADER_DATASETS and model == "MLP":
            # In-VRAM MLP: skip NeighborLoader entirely. data.x lives on CPU for
            # loader datasets, so index on CPU and move only the selected slice
            # to GPU — indexing a CPU tensor with a CUDA mask raises RuntimeError.
            x_train = data.x[masks['train_mask']].to(device, non_blocking=True)
            y_train = data.y[masks['train_mask']].to(device, non_blocking=True)
            x_val = data.x[masks['val_mask']].to(device, non_blocking=True)
            y_val = data.y[masks['val_mask']].to(device, non_blocking=True)
            try:
                _best_wts, best_pr_auc = train_and_validate_in_vram(
                    model_wrapper, x_train, y_train, x_val, y_val,
                    n_epochs, MLP_IN_VRAM_BATCH_SIZE, trial=trial
                )
            finally:
                del x_train, y_train, x_val, y_val
        elif dataset_name in LOADER_DATASETS and batch_size is not None:
            loader_kwargs = neighbor_loader_kwargs()
            train_loader = NeighborLoader(data, num_neighbors=num_neighbors,
                                          batch_size=batch_size, input_nodes=train_mask,
                                          shuffle=True, is_sorted=True, **loader_kwargs)
            val_loader = NeighborLoader(data, num_neighbors=num_neighbors,
                                        batch_size=batch_size, input_nodes=masks['val_mask'],
                                        is_sorted=True, **loader_kwargs)
            _best_wts, best_pr_auc = train_and_validate_with_loader(
                model_wrapper, train_loader, val_loader, n_epochs, trial=trial
            )
        elif dataset_name in FULL_BATCH_DATASETS:
            _best_wts, best_pr_auc = train_and_validate(
                model_wrapper, data, train_val_masks, n_epochs, dataset_name, trial=trial
            )
        else:
            raise ValueError(f"Unsupported dataset for wrapper model tuning: {dataset_name}")

        return best_pr_auc

    elif model in SKLEARN_MODELS:
        if dataset_name == "Elliptic":
            train_mask = masks['train_perf_eval_mask']
            val_mask = masks['val_perf_eval_mask']
        else:
            train_mask = masks['train_mask']
            val_mask = masks['val_mask']

        train_x = data.x[train_mask].cpu().numpy()
        train_y = data.y[train_mask].cpu().numpy()
        val_x = data.x[val_mask].cpu().numpy()
        val_y = data.y[val_mask].cpu().numpy()

        model_instance.fit(train_x, train_y)

        # Use predicted positive-class probabilities (or scaled decision function
        # for SVM) to compute PR-AUC as the unified Optuna objective.
        if hasattr(model_instance, 'predict_proba'):
            val_scores = model_instance.predict_proba(val_x)[:, 1]
        elif hasattr(model_instance, 'decision_function'):
            dfunc = model_instance.decision_function(val_x)
            denom = (dfunc.max() - dfunc.min())
            val_scores = (dfunc - dfunc.min()) / denom if denom > 0 else np.zeros_like(dfunc)
        else:
            val_scores = model_instance.predict(val_x)

        from sklearn.metrics import average_precision_score
        val_y_np = val_y if isinstance(val_y, np.ndarray) else np.asarray(val_y)
        if len(np.unique(val_y_np)) < 2:
            return 0.0
        pr_auc = average_precision_score(val_y_np, val_scores)
        return float(pr_auc)

    else:
        raise ValueError(f"Unknown model type: {model}")
