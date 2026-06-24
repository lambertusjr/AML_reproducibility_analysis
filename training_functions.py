import torch
import gc
import optuna
from utilities import ram_is_critical, check_ram_usage, vram_is_critical, check_vram_usage, cuda_context_healthy


def _periodic_memory_cleanup(epoch):
    if (epoch + 1) % 25 == 0:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        if ram_is_critical(threshold=0.85):
            usage_pct, avail_gb = check_ram_usage()
            print(f"WARNING: RAM usage {usage_pct:.1f}% at epoch {epoch+1}, "
                  f"only {avail_gb:.1f} GB available. System may start paging to SSD.")
        if torch.cuda.is_available() and vram_is_critical(threshold=0.80):
            vram_frac, vram_free = check_vram_usage()
            print(f"WARNING: VRAM usage {vram_frac*100:.1f}% at epoch {epoch+1}, "
                  f"only {vram_free:.2f} GB free.")


def train_and_validate_with_loader(model_wrapper, train_loader, val_loader, num_epochs, trial=None):
    """
    Train for ``num_epochs`` epochs with no early stopping. If ``val_loader`` is
    None, no per-epoch validation runs and the final-epoch weights are returned
    (used for the final-test fit on train ∪ val). Otherwise the per-epoch
    validation PR-AUC drives best-weights checkpointing. If an Optuna ``trial``
    is supplied, the val PR-AUC is reported and the trial is pruned when asked.
    """
    best_pr_auc = -1.0
    best_model_wts = None

    for epoch in range(num_epochs):
        try:
            model_wrapper.train_step_loader(train_loader)

            if val_loader is not None:
                _, _val_f1, val_pr_auc = model_wrapper.evaluate_loader_mini(val_loader)
                if val_pr_auc > best_pr_auc:
                    best_pr_auc, best_model_wts = update_best_weights(
                        model_wrapper.model, best_pr_auc, val_pr_auc, best_model_wts
                    )
                if trial is not None:
                    trial.report(val_pr_auc, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()
                print(f"CUDA OOM at epoch {epoch+1}. Pruning trial.")
                if not cuda_context_healthy():
                    raise RuntimeError(
                        f"CUDA context corrupted after OOM at epoch {epoch+1}. "
                        "Restart the kernel to avoid access violations."
                    )
            raise

        _periodic_memory_cleanup(epoch)

    if val_loader is None:
        final_wts = {k: v.cpu().clone().detach()
                     for k, v in model_wrapper.model.state_dict().items()}
        return final_wts, float('nan')

    return best_model_wts, best_pr_auc


def train_and_validate_in_vram(model_wrapper, x_train, y_train, x_val, y_val,
                               num_epochs, batch_size, trial=None):
    """
    Train a graph-agnostic model (MLP) directly on GPU-resident feature tensors.
    ``x_train`` / ``y_train`` (and optionally ``x_val`` / ``y_val``) must already
    live on the model's device. Skips NeighborLoader entirely (the sampler does
    CPU graph traversal the MLP never uses). If ``x_val`` is None, final-epoch
    weights are returned.
    """
    best_pr_auc = -1.0
    best_model_wts = None

    for epoch in range(num_epochs):
        try:
            model_wrapper.train_step_in_vram(x_train, y_train, batch_size)

            if x_val is not None:
                _, _val_f1, val_pr_auc = model_wrapper.mini_eval_in_vram(x_val, y_val, batch_size)
                if val_pr_auc > best_pr_auc:
                    best_pr_auc, best_model_wts = update_best_weights(
                        model_wrapper.model, best_pr_auc, val_pr_auc, best_model_wts
                    )
                if trial is not None:
                    trial.report(val_pr_auc, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()
                print(f"CUDA OOM at epoch {epoch+1}. Pruning trial.")
                if not cuda_context_healthy():
                    raise RuntimeError(
                        f"CUDA context corrupted after OOM at epoch {epoch+1}. "
                        "Restart the kernel to avoid access violations."
                    )
            raise

        _periodic_memory_cleanup(epoch)

    if x_val is None:
        final_wts = {k: v.cpu().clone().detach()
                     for k, v in model_wrapper.model.state_dict().items()}
        return final_wts, float('nan')

    return best_model_wts, best_pr_auc


def train_and_validate(model_wrapper, data, masks, num_epochs, dataset_name, trial=None):
    """
    Full-batch training with no early stopping. If the validation mask is None,
    final-epoch weights are returned; otherwise validation PR-AUC drives
    best-weights checkpointing. Elliptic uses its perf-eval masks; any other
    full-batch dataset uses the generic train/val masks.
    """
    best_pr_auc = -1.0
    best_model_wts = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    masks = {k: (v.to(device) if v is not None else None) for k, v in masks.items()}

    is_elliptic = (dataset_name == "Elliptic")
    if is_elliptic:
        elliptic_training_masks = [masks['train_mask'], masks['train_perf_eval_mask']]
        if masks.get('val_perf_eval_mask') is not None:
            elliptic_validation_masks = [masks['val_mask'], masks['val_perf_eval_mask']]
        else:
            elliptic_validation_masks = None

    for epoch in range(num_epochs):
        reported_pr_auc = None
        if is_elliptic:
            model_wrapper.train_step_elliptic(data, elliptic_training_masks)
            if elliptic_validation_masks is not None:
                _, _val_f1, val_pr_auc = model_wrapper.mini_eval_elliptic(
                    data, elliptic_validation_masks
                )
                if val_pr_auc > best_pr_auc:
                    best_pr_auc, best_model_wts = update_best_weights(
                        model_wrapper.model, best_pr_auc, val_pr_auc, best_model_wts
                    )
                reported_pr_auc = val_pr_auc
        else:
            # Generic full-batch path for any non-Elliptic full-batch dataset.
            model_wrapper.train_step_full(data, masks['train_mask'])
            if masks.get('val_mask') is not None:
                _, _val_f1, val_pr_auc = model_wrapper.mini_eval_full(data, masks['val_mask'])
                if val_pr_auc > best_pr_auc:
                    best_pr_auc, best_model_wts = update_best_weights(
                        model_wrapper.model, best_pr_auc, val_pr_auc, best_model_wts
                    )
                reported_pr_auc = val_pr_auc

        if trial is not None and reported_pr_auc is not None:
            trial.report(reported_pr_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        _periodic_memory_cleanup(epoch)

    val_present = (is_elliptic and elliptic_validation_masks is not None) or \
                  (not is_elliptic and masks.get('val_mask') is not None)
    if not val_present:
        final_wts = {k: v.cpu().clone().detach()
                     for k, v in model_wrapper.model.state_dict().items()}
        return final_wts, float('nan')

    return best_model_wts, best_pr_auc


def update_best_weights(model, best_value, current_value, best_model_wts=None):
    if current_value > best_value:
        best_value = current_value
        best_model_wts = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
    return best_value, best_model_wts
