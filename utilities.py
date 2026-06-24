import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import os
import json
import psutil


def configure_gpu_memory_limits(fraction=0.95):
    """
    Cap the fraction of GPU memory PyTorch may use, to prevent spillover into
    system RAM.

    Note: PYTORCH_CUDA_ALLOC_CONF must be set via os.environ BEFORE any torch
    import (done at the top of main.py); setting it here would have no effect.
    """
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
        print(f"GPU memory limited to {fraction*100}% of available memory")

        alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'not set')
        print(f"PYTORCH_CUDA_ALLOC_CONF: {alloc_conf}")

    ram = psutil.virtual_memory()
    print(f"System RAM: {ram.total / (1024**3):.1f} GB total, "
          f"{ram.available / (1024**3):.1f} GB available, "
          f"{ram.percent}% used")


def check_ram_usage():
    """
    Current RAM usage.

    Returns:
        tuple: (usage_percent, available_gb)
    """
    ram = psutil.virtual_memory()
    return ram.percent, ram.available / (1024**3)


def ram_is_critical(threshold=0.85):
    """True if RAM usage exceeds ``threshold`` (default 85%)."""
    usage_percent, _ = check_ram_usage()
    return usage_percent > (threshold * 100)


def check_vram_usage():
    """
    Current GPU VRAM usage.

    Returns:
        tuple: (usage_fraction, free_gb). Uses memory_reserved (caching
        allocator pool) rather than memory_allocated, because nvidia-smi and the
        OS see reserved memory as consumed.
    """
    if not torch.cuda.is_available():
        return 0.0, float('inf')
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    usage_fraction = reserved / total
    free_gb = (total - reserved) / (1024**3)
    return usage_fraction, free_gb


def vram_is_critical(threshold=0.90):
    """True if VRAM usage exceeds ``threshold`` (default 90%)."""
    usage_fraction, _ = check_vram_usage()
    return usage_fraction > threshold


def cuda_context_healthy():
    """
    Verify the CUDA context is still usable after a caught OOM/error via a small
    allocation + sync. Returns True if CUDA is unavailable (CPU-only) or the
    context is healthy, False if it appears corrupted.
    """
    if not torch.cuda.is_available():
        return True
    try:
        torch.cuda.synchronize()
        probe = torch.zeros(1, device='cuda')
        del probe
        torch.cuda.synchronize()
        return True
    except RuntimeError as e:
        print(f"CUDA context health check FAILED: {e}")
        print("The GPU context is likely corrupted. Further CUDA operations may "
              "cause access violations. Consider restarting the kernel.")
        return False


def _cache_path(dataset_name, model_name):
    """Per-combo batch-size cache filename (one file per combo avoids races)."""
    return f'batch_size_cache_{dataset_name}_{model_name}.json'


def save_batch_size_by_phase(dataset_name, model_name, batch_size, phase='tuning', cache_file=None):
    """Save the optimal batch size for a (dataset, model, phase) combination."""
    if cache_file is None:
        cache_file = _cache_path(dataset_name, model_name)

    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not read {cache_file}, creating new cache")

    key = f"{dataset_name}_{model_name}_{phase}"
    cache[key] = batch_size

    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"Saved {phase} batch size {batch_size} for {dataset_name}_{model_name}")


def load_batch_size_by_phase(dataset_name, model_name, phase='tuning', cache_file=None):
    """Load the cached batch size for a (dataset, model, phase) combination, or None."""
    if cache_file is None:
        cache_file = _cache_path(dataset_name, model_name)

    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, 'r') as f:
            cache = json.load(f)

        key = f"{dataset_name}_{model_name}_{phase}"
        batch_size = cache.get(key)

        if batch_size is not None:
            print(f"Loaded cached {phase} batch size {batch_size} for {dataset_name}_{model_name}")
            return batch_size
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load from {cache_file}: {e}")

    return None


def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            alpha_val = float(alpha)
            if not (0.0 <= alpha_val <= 1.0):
                raise ValueError("alpha float must lie in [0, 1]")
            self.alpha = torch.tensor([alpha_val, 1.0 - alpha_val], dtype=torch.float32)
        elif isinstance(alpha, (list, tuple, torch.Tensor)):
            self.alpha = torch.as_tensor(alpha, dtype=torch.float32)
        else:
            raise TypeError("alpha must be None, float, sequence, or torch.Tensor")

        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.ndim != 1:
                raise ValueError("alpha tensor must be 1-dimensional")
            if torch.any(self.alpha < 0):
                raise ValueError("alpha tensor must be non-negative")
            if self.alpha.sum() == 0:
                raise ValueError("alpha tensor must have positive sum")
            self.alpha = self.alpha / self.alpha.sum()

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logits = inputs.float()
        targets = targets.long()

        num_classes = logits.shape[-1]

        if targets.min() < 0 or targets.max() >= num_classes:
            raise ValueError(
                f"Target values must be in range [0, {num_classes-1}], "
                f"but got min={targets.min().item()}, max={targets.max().item()}"
            )

        if self.alpha is not None and len(self.alpha) != num_classes:
            raise ValueError(
                f"Alpha dimension ({len(self.alpha)}) must match number of classes ({num_classes})"
            )

        # log_softmax + nll_loss for numerical stability.
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')

        pt = torch.exp(-ce_loss).clamp(min=1e-7, max=1.0 - 1e-7)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha[targets]  # per-class weights
            focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def extract_data_information(data):
    """Extract masks and rebuild a clean Data object without extra attributes."""
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    y = data.y
    x = data.x
    edge_index = data.edge_index
    del data
    new_data = Data(x=x, edge_index=edge_index, y=y)
    return new_data, train_mask, val_mask, test_mask


def verify_xgboost_gpu_support():
    """Print XGBoost/CUDA configuration and run a tiny GPU fit to verify support."""
    print("\n" + "=" * 80)
    print("XGBoost GPU Configuration Check")
    print("=" * 80)

    try:
        import xgboost as xgb
        print(f"XGBoost version: {xgb.__version__}")
    except ImportError:
        print("XGBoost is not installed")
        return False

    if torch.cuda.is_available():
        print(f"PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("PyTorch CUDA not available")
        return False

    try:
        from xgboost import XGBClassifier
        print("\nTesting XGBoost GPU functionality...")

        test_X = np.random.rand(100, 10)
        test_y = np.random.randint(0, 2, 100)

        test_model = XGBClassifier(
            n_estimators=10,
            tree_method='hist',
            device='cuda',
            eval_metric='logloss'
        )
        test_model.fit(test_X, test_y, verbose=False)

        print("XGBoost GPU test successful.")
        print(f"  Device used: {test_model.get_params()['device']}")
        print(f"  Tree method: {test_model.get_params()['tree_method']}")
        print("=" * 80 + "\n")
        return True

    except Exception as e:
        print(f"XGBoost GPU test failed: {str(e)}")
        print("Troubleshooting: ensure an xgboost build with GPU support, a CUDA "
              "toolkit matching the PyTorch CUDA version, and tree_method='hist'.")
        print("=" * 80 + "\n")
        return False


def extract_and_remove_masks(data):
    """
    Extract the mask attributes from a Data object, remove them from the object,
    and return (data, masks_dict). Missing masks are set to None.
    """
    mask_keys = [
        'train_mask', 'val_mask', 'test_mask',
        'train_perf_eval_mask', 'val_perf_eval_mask', 'test_perf_eval_mask'
    ]

    extracted_masks = {}
    for key in mask_keys:
        if hasattr(data, key):
            extracted_masks[key] = getattr(data, key)
            delattr(data, key)
        else:
            extracted_masks[key] = None

    return data, extracted_masks
