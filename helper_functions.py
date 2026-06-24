import sys
import torch
import numpy as np
from torchmetrics.classification import BinaryPrecisionRecallCurve
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, precision_score,
                             recall_score, roc_auc_score, auc, average_precision_score,
                             matthews_corrcoef)
import matplotlib.pyplot as plt
import pickle
import os
import optuna

from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
import gc
import time
from contextlib import contextmanager
from utilities import cuda_context_healthy
from torch_geometric.loader import NeighborLoader

from config import N_TRIALS


def calculate_pr_metrics_batched(probs, labels, chunk_size=10000):
    # thresholds=None calculates the exact curve (more RAM); a fixed bin count
    # would cap RAM at the cost of resolution.
    pr_curve = BinaryPrecisionRecallCurve(thresholds=None).cpu()

    n_samples = probs.size(0)

    with torch.no_grad():
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            # Slice and move to CPU immediately: .detach() breaks the graph,
            # .cpu() frees VRAM.
            prob_chunk = probs[i:end].detach().cpu()
            label_chunk = labels[i:end].detach().cpu()
            pr_curve.update(prob_chunk, label_chunk)

    precision, recall, thresholds = pr_curve.compute()
    return precision, recall, thresholds


def save_pr_artifacts(precision, recall, thresholds, filename_prefix):
    """Take pre-calculated PR metrics and save the curve data + image to disk."""
    pr_auc = auc(recall.numpy(), precision.numpy())

    np.savez_compressed(
        f"{filename_prefix}_pr_data.npz",
        precision=precision.numpy(),
        recall=recall.numpy(),
        thresholds=thresholds.numpy(),
        auc=pr_auc
    )

    try:
        fig, ax = plt.subplots()
        ax.plot(recall.numpy(), precision.numpy(), label=f'PRAUC = {pr_auc:.4f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend()
        fig.savefig(f"{filename_prefix}_pr_curve.png", dpi=300)
    finally:
        plt.close(fig)  # prevent memory leak

    return pr_auc


def save_metrics_to_pickle(metrics_dict, filename):
    """Save a metrics dictionary to a pickle file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(metrics_dict, f)
    print(f"Saved metrics to {filename}")


def save_dataframe_to_pickle(df, filename):
    """Save a pandas DataFrame to a pickle file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(df, f)
    print(f"Saved DataFrame to {filename}")


def load_pickle(filename):
    """Load data from a pickle file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def calculate_metrics(y_true, y_pred, y_pred_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_illicit = precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)  # illicit is class 1
    recall = recall_score(y_true, y_pred, average='weighted')
    recall_illicit = recall_score(y_true, y_pred, pos_label=1, average='binary')  # illicit is class 1
    f1 = f1_score(y_true, y_pred, average='weighted')
    f1_illicit = f1_score(y_true, y_pred, pos_label=1, average='binary')  # illicit is class 1

    prob_positive = y_pred_prob[:, 1] if y_pred_prob.ndim == 2 else y_pred_prob
    roc_auc = roc_auc_score(y_true, prob_positive)  # class 1 is the positive class
    pr_auc = np.nan  # PR-AUC is computed from the saved PR curve in testing.py

    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'precision_illicit': precision_illicit,
        'recall': recall,
        'recall_illicit': recall_illicit,
        'f1': f1,
        'f1_illicit': f1_illicit,
        'roc_auc': roc_auc,
        'PRAUC': pr_auc,
        'MCC': mcc,
        'kappa': kappa,
    }

    return metrics


def balanced_class_weights(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    Compute inverse-frequency class weights (sum to 1) for 1-D integer labels.
    Unlabelled entries (label < 0) are ignored.
    """
    if labels.ndim != 1:
        labels = labels.view(-1)
    labels = labels.detach().cpu()
    valid = labels >= 0
    if not torch.any(valid):
        return torch.ones(num_classes, dtype=torch.float32) / float(num_classes)
    filtered = labels[valid].to(torch.long)
    counts = torch.bincount(filtered, minlength=num_classes).clamp_min(1)
    inv = (1.0 / counts.float())
    inv = inv / inv.sum()
    return inv


def _get_model_instance(trial, model, data, device, train_mask=None):
    """
    Suggest hyperparameters and instantiate a model based on its name. The
    search ranges defined here are the canonical source for the README's
    hyperparameter-search-space table.
    """
    if model == 'MLP':
        from models import MLP
        hidden_units = trial.suggest_int('hidden_units', 32, 256)
        dropout_1 = trial.suggest_float('dropout_1', 0.0, 0.7)
        dropout_2 = trial.suggest_float('dropout_2', 0.0, 0.7)
        return MLP(num_node_features=data.num_node_features, num_classes=2,
                   hidden_units=hidden_units, dropout_1=dropout_1, dropout_2=dropout_2)

    elif model == 'SVM':
        C = trial.suggest_float('C', 0.01, 100.0, log=True)
        return SGDClassifier(
            loss='hinge',
            alpha=1.0 / (C * data.num_nodes),  # convert C to alpha
            penalty='l2',
            max_iter=1000,
            tol=1e-3,
            class_weight='balanced',
            random_state=42
        )

    elif model == 'XGB':
        max_depth = trial.suggest_int('max_depth', 5, 15)
        Gamma_XGB = trial.suggest_float('Gamma_XGB', 0, 5)
        n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
        learning_rate_XGB = trial.suggest_float('learning_rate_XGB', 0.001, 0.3, log=True)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
        return XGBClassifier(
            eval_metric='logloss',
            scale_pos_weight=calculate_scale_pos_weight(data, train_mask),
            learning_rate=learning_rate_XGB,
            max_depth=max_depth,
            n_estimators=n_estimators,
            colsample_bytree=colsample_bytree,
            gamma=Gamma_XGB,
            subsample=subsample,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            tree_method='hist',
            device="cpu"
        )

    elif model == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
        max_depth = trial.suggest_int('max_depth', 5, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                      max_features=max_features, class_weight='balanced', n_jobs=-1)

    elif model == 'GCN':
        from models import GCN
        hidden_units = trial.suggest_int('hidden_units', 32, 256)
        dropout = trial.suggest_float('dropout', 0.0, 0.7)
        n_layers = trial.suggest_int('n_layers', 1, 3)
        return GCN(num_node_features=data.x.shape[1], num_classes=2,
                   hidden_units=hidden_units, dropout=dropout, n_layers=n_layers)

    elif model == 'GAT':
        from models import GAT
        hidden_units = trial.suggest_int('hidden_units', 32, 256)
        num_heads = trial.suggest_int('num_heads', 1, 8)
        dropout_1 = trial.suggest_float('dropout_1', 0.0, 0.7)
        dropout_2 = trial.suggest_float('dropout_2', 0.0, 0.7)
        # Backward compat: old trials lack feature_dropout, default to dropout_1
        try:
            feature_dropout = trial.suggest_float('feature_dropout', 0.0, 0.7)
        except Exception:
            feature_dropout = dropout_1
        n_layers = trial.suggest_int('n_layers', 1, 3)
        return GAT(num_node_features=data.x.shape[1], num_classes=2,
                   hidden_units=hidden_units, num_heads=num_heads,
                   dropout_1=dropout_1, dropout_2=dropout_2,
                   feature_dropout=feature_dropout, n_layers=n_layers)

    elif model == 'GIN':
        from models import GIN
        hidden_units = trial.suggest_int('hidden_units', 32, 256)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        n_layers = trial.suggest_int('n_layers', 1, 3)
        return GIN(num_node_features=data.x.shape[1], num_classes=2,
                   hidden_units=hidden_units, dropout=dropout, n_layers=n_layers)

    else:
        raise ValueError(f"Unknown model: {model}")


def calculate_scale_pos_weight(data, train_mask):
    """Compute scale_pos_weight (neg/pos) on the training split only."""
    if train_mask is None:
        raise ValueError("train_mask must be provided to calculate_scale_pos_weight to prevent data leakage.")

    y_train = data.y[train_mask].cpu().numpy()
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    return float(neg) / float(pos)


def check_study_existence(model_name, data_for_optimization):
    """
    Check whether an Optuna study exists and has run the required number of
    trials. The target is ``config.N_TRIALS`` (200) for every model. Incomplete
    studies are not deleted — ``hyperparameter_tuning`` resumes them, running
    only the remaining trials.

    Returns True if the study has >= N_TRIALS settled trials, else False.
    """
    required = N_TRIALS

    study_name = f'{model_name}_optimization on {data_for_optimization} dataset'
    storage_url = f'sqlite:///optimization_results_on_{data_for_optimization}_{model_name}.db'

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)

        # A PRUNED trial is a finished attempt and counts toward the target,
        # matching the "settled" semantics used by hyperparameter_tuning.
        settled_states = {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        }
        num_completed = len([t for t in study.trials if t.state in settled_states])

        if num_completed >= required:
            print(f"Study '{study_name}' complete: {num_completed}/{required} trials done.")
            return True
        else:
            remaining = required - num_completed
            print(f"Study '{study_name}' incomplete: {num_completed}/{required} trials done, "
                  f"{remaining} remaining.")
            return False

    except KeyError:
        print(f"Study '{study_name}' not found.")
        return False


def run_trial_with_cleanup(trial_func, model_name, *args, **kwargs):
    """
    Run a trial function safely: no_grad for CPU-based models, plus aggressive
    GPU/CPU memory reclamation after each trial.
    """
    try:
        with inference_mode_if_needed(model_name):
            result = trial_func(*args, **kwargs)
        return result
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        gc.collect()
        gc.collect()  # twice, to catch circular references

        if torch.cuda.is_available() and not cuda_context_healthy():
            raise RuntimeError(
                "CUDA context corrupted after trial cleanup. "
                "Restart the kernel to avoid access violations."
            )


@contextmanager
def inference_mode_if_needed(model_name: str):
    """Disable gradient tracking for CPU-based (sklearn) models."""
    if model_name in ["SVM", "XGB", "RF"]:
        with torch.no_grad():
            yield
    else:
        yield


_SAMPLER_BACKEND_LOGGED = False


def _log_sampler_backend_once():
    global _SAMPLER_BACKEND_LOGGED
    if _SAMPLER_BACKEND_LOGGED:
        return
    _SAMPLER_BACKEND_LOGGED = True
    try:
        import pyg_lib
        print(f"[sampler] pyg-lib {pyg_lib.__version__} detected -> C++ neighbor sampler ACTIVE")
    except (ImportError, OSError):
        # OSError covers the GLIBC-too-old case where the wheel installs but
        # libpyg.so fails to dlopen at import time.
        print("[sampler] pyg-lib not usable -> falling back to torch-sparse sampler")


def ensure_edge_index_column_sorted(data, name=None):
    """
    Validate that data.edge_index is sorted by destination column (CSC layout)
    so NeighborLoader(is_sorted=True) is safe. If an on-disk cache predates the
    column-sort change, re-sort in memory with a warning rather than feeding the
    sampler an incorrect assumption.
    """
    ei = data.edge_index
    col = ei[1]
    is_sorted = bool(torch.all(col[1:] >= col[:-1]).item())
    if is_sorted:
        return data
    from torch_geometric.utils import sort_edge_index
    tag = f" [{name}]" if name else ""
    print(f"[sampler]{tag} edge_index is not column-sorted -> resorting in memory. "
          f"Delete Datasets/.../processed/*.pt and rerun pre_process_datasets.py "
          f"to make this persistent.")
    data.edge_index = sort_edge_index(
        ei, num_nodes=data.num_nodes, sort_by_row=False,
    )
    return data


def neighbor_loader_kwargs(num_workers=None):
    """
    Standard NeighborLoader kwargs for real training/eval (not the batch-size
    probe). Keeps workers alive across epochs and pages CPU samples into pinned
    memory for faster host-to-device copies. Worker count adapts to available
    RAM (each worker forks the graph and holds pinned prefetch buffers).
    """
    _log_sampler_backend_once()
    if num_workers is None:
        if sys.platform == 'win32':
            num_workers = 0
        else:
            from utilities import check_ram_usage
            cpu_cap = (os.cpu_count() or 1)
            _, avail_gb = check_ram_usage()
            if avail_gb > 24:
                num_workers = min(6, cpu_cap)
            elif avail_gb > 12:
                num_workers = min(4, cpu_cap)
            else:
                num_workers = min(2, cpu_cap)
    kwargs = {'num_workers': num_workers}
    if num_workers > 0:
        kwargs['persistent_workers'] = True
    if torch.cuda.is_available():
        kwargs['pin_memory'] = True
    return kwargs


def find_optimal_batch_size(model_builder, data, device, train_mask, num_neighbors=[10, 5],
                            dataset_name=None, model_name=None, phase='tuning'):
    """
    Find the largest NeighborLoader batch size that fits without paging to
    system RAM (exponential search to an upper bound, then binary search).
    Results are cached per (dataset, model, phase).
    """
    if dataset_name and model_name:
        from utilities import load_batch_size_by_phase
        cached_batch_size = load_batch_size_by_phase(dataset_name, model_name, phase=phase)
        if cached_batch_size is not None:
            return cached_batch_size

    print(f"Searching for optimal {phase} batch size...")

    if torch.cuda.is_available():
        total_gpu_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_fraction = 0.90  # leave headroom to avoid OOM during training
        print(f"Running in {phase.upper()} mode: will use up to 90% of GPU memory")
        max_memory_limit = int(total_gpu_memory * reserved_fraction)
        print(f"GPU memory limit: {max_memory_limit / (1024**3):.2f} GB")
    else:
        max_memory_limit = None

    num_train_nodes = int(train_mask.sum()) if train_mask.dtype == torch.bool else len(train_mask)
    low = 16384  # start high for better GPU utilisation
    # Cap at 800k. Beyond this, NeighborLoader sampling cost dominates CPU RAM
    # (a 128k batch with [10,5] fanout samples ~8M nodes; doubling the batch
    # doubles the working set but barely moves GPU throughput).
    high = min(num_train_nodes, 800000)
    optimal = 65536  # higher safe default

    from utilities import ram_is_critical, check_ram_usage, vram_is_critical, check_vram_usage

    def test_batch_size(batch_size):
        # Guard: check RAM and VRAM before attempting this batch size.
        usage_pct, avail_gb = check_ram_usage()
        if ram_is_critical(threshold=0.75):
            print(f"Batch size {batch_size} skipped: RAM already at {usage_pct:.1f}% "
                  f"({avail_gb:.1f} GB available) before test started.")
            return False

        if torch.cuda.is_available():
            vram_frac, vram_free = check_vram_usage()
            if vram_is_critical(threshold=0.80):
                print(f"Batch size {batch_size} skipped: VRAM already at {vram_frac*100:.1f}% "
                      f"({vram_free:.2f} GB free) before test started.")
                return False

        try:
            model = model_builder().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()

            loader = NeighborLoader(
                data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                input_nodes=train_mask,
                shuffle=True,
                is_sorted=True,
                **neighbor_loader_kwargs()
            )

            model.train()
            steps = 0
            ram_exceeded = False
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)

                batch_size_actual = batch.batch_size
                out_sliced = out[:batch_size_actual]
                y_sliced = batch.y[:batch_size_actual]

                loss = criterion(out_sliced, y_sliced)
                loss.backward()
                optimizer.step()

                steps += 1

                # Check RAM every step so we abort as soon as paging would begin.
                usage_pct, avail_gb = check_ram_usage()
                if ram_is_critical(threshold=0.75):
                    ram_exceeded = True
                    print(f"Batch size {batch_size} rejected at step {steps}: "
                          f"RAM usage {usage_pct:.1f}% ({avail_gb:.1f} GB available). "
                          f"Reducing to prevent SSD paging.")
                    break

                if torch.cuda.is_available():
                    vram_frac, vram_free = check_vram_usage()
                    if vram_is_critical(threshold=0.80):
                        ram_exceeded = True
                        print(f"Batch size {batch_size} rejected at step {steps}: "
                              f"VRAM usage {vram_frac*100:.1f}% ({vram_free:.2f} GB free). "
                              f"Reducing to prevent GPU OOM.")
                        break

                if steps >= 10:  # 10 batches give a representative memory profile
                    break

            del model, optimizer, criterion, loader
            if steps > 0:
                del batch, out, out_sliced, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return not ram_exceeded

        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                print(f"Batch size {batch_size} failed with GPU OOM/CUDA error: {e}")
            else:
                print(f"Batch size {batch_size} failed with RuntimeError: {e}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not cuda_context_healthy():
                raise RuntimeError(
                    "CUDA context corrupted after OOM during batch size search. "
                    "Restart the kernel to avoid access violations."
                )
            return False
        except Exception as e:
            print(f"Batch size {batch_size} failed with error: {type(e).__name__}: {e}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            return False

    # Force a GC + cache clear before the search so residual memory from dataset
    # loading doesn't make the first batch size fail.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Exponential search for an upper bound.
    current = low
    max_safe = None

    while current <= high:
        print(f"Testing batch size: {current}")
        if test_batch_size(current):
            max_safe = current
            current *= 2
        else:
            high = current
            break

    if max_safe is None:
        for fallback in [16384, 8192, 4096, 2048, 1024, 512]:
            print(f"Fallback testing batch size: {fallback}")
            if test_batch_size(fallback):
                max_safe = fallback
                break
        if max_safe is None:
            print("WARNING: All batch sizes failed. Using minimum fallback of 512.")
            return 512

    # 2. Binary search between max_safe and high (which failed).
    low = max_safe
    while low < high - 2048:  # granularity of 2048
        mid = (low + high) // 2
        print(f"Binary search testing: {mid}")
        if test_batch_size(mid):
            low = mid
        else:
            high = mid

    # Use a percentage of the max found size (threshold already adds margin).
    optimal = int(low * (0.97 if phase == 'evaluation' else 0.95))
    print(f"Optimal {phase} batch size found: {optimal}")

    if dataset_name and model_name:
        from utilities import save_batch_size_by_phase
        save_batch_size_by_phase(dataset_name, model_name, optimal, phase=phase)

    return optimal
