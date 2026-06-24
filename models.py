import math
import os
import time
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv
from torch_geometric.nn.norm import GraphNorm
from helper_functions import calculate_metrics
from sklearn.metrics import f1_score, average_precision_score
from torchmetrics.classification import BinaryF1Score

_PROFILE_DONE = False  # set to True after the first profiled epoch prints


def _safe_average_precision(y_true, y_score):
    # average_precision_score raises if only one class is present in y_true.
    # Return 0.0 in that degenerate case so the metric remains comparable.
    import numpy as np
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


# %% ModelWrapper

class ModelWrapper:
    def __init__(self, model, optimiser, criterion):
        self.model = model
        self.optimiser = optimiser
        self.criterion = criterion

    def train_step_loader(self, loader):
        global _PROFILE_DONE
        profile = (os.environ.get('AML_PROFILE') == '1') and not _PROFILE_DONE
        sample_t = h2d_t = forward_t = backward_t = bookkeep_t = 0.0
        n_profiled = 0
        cuda_sync = torch.cuda.is_available()

        self.model.train()
        total_loss = 0
        device = next(self.model.parameters()).device
        all_preds, all_labels = [], []

        if profile:
            t_loop_start = time.perf_counter()
            it = iter(loader)
            while True:
                if cuda_sync: torch.cuda.synchronize()
                t_pre_next = time.perf_counter()
                try:
                    batch = next(it)
                except StopIteration:
                    break
                if cuda_sync: torch.cuda.synchronize()
                t_post_next = time.perf_counter()

                batch = batch.to(device)
                if cuda_sync: torch.cuda.synchronize()
                t_h2d = time.perf_counter()

                self.optimiser.zero_grad()
                out = self.model(batch)
                batch_size = batch.batch_size
                out_sliced = out[:batch_size]
                y_sliced = batch.y[:batch_size]
                loss = self.criterion(out_sliced, y_sliced)
                if cuda_sync: torch.cuda.synchronize()
                t_fwd = time.perf_counter()

                loss.backward()
                self.optimiser.step()
                if cuda_sync: torch.cuda.synchronize()
                t_bwd = time.perf_counter()

                total_loss += float(loss.detach())
                pred = out_sliced.argmax(dim=1)
                all_preds.append(pred.cpu())
                all_labels.append(y_sliced.cpu())
                del batch, out, out_sliced, y_sliced, loss, pred
                if cuda_sync: torch.cuda.synchronize()
                t_end = time.perf_counter()

                sample_t   += (t_post_next - t_pre_next)
                h2d_t      += (t_h2d - t_post_next)
                forward_t  += (t_fwd - t_h2d)
                backward_t += (t_bwd - t_fwd)
                bookkeep_t += (t_end - t_bwd)
                n_profiled += 1
                if n_profiled >= 50:
                    break

            total = time.perf_counter() - t_loop_start
            print(f"\n[AML_PROFILE] First {n_profiled} batches of first epoch "
                  f"(num_workers={getattr(loader, 'num_workers', '?')}, "
                  f"batch_size={getattr(loader, 'batch_size', '?')}):")
            print(f"  sample (next): {sample_t   * 1000 / n_profiled:7.1f} ms/batch")
            print(f"  H2D copy     : {h2d_t      * 1000 / n_profiled:7.1f} ms/batch")
            print(f"  forward+loss : {forward_t  * 1000 / n_profiled:7.1f} ms/batch")
            print(f"  backward+opt : {backward_t * 1000 / n_profiled:7.1f} ms/batch")
            print(f"  bookkeep(D2H): {bookkeep_t * 1000 / n_profiled:7.1f} ms/batch")
            print(f"  total /batch : {total      * 1000 / n_profiled:7.1f} ms")
            measured = sample_t + h2d_t + forward_t + backward_t + bookkeep_t
            if measured > 0:
                print(f"  -> sample share : {100 * sample_t / measured:.0f}%")
                print(f"     compute share: {100 * (forward_t + backward_t) / measured:.0f}%")
                print(f"     overhead     : {100 * (h2d_t + bookkeep_t) / measured:.0f}%")
            _PROFILE_DONE = True

            if all_labels:
                import numpy as _np
                y_true = torch.cat(all_labels).numpy()
                y_pred = torch.cat(all_preds).numpy()
                if len(_np.unique(y_true)) > 1:
                    f1_illicit = f1_score(y_true, y_pred, pos_label=1, average='binary')
                else:
                    f1_illicit = 0.0
                return total_loss / n_profiled, f1_illicit
            return 0.0, 0.0

        for batch in loader:
            batch = batch.to(device)
            self.optimiser.zero_grad()

            out = self.model(batch)
            # Slice to get only target nodes (first batch_size nodes in NeighborLoader)
            batch_size = batch.batch_size
            out_sliced = out[:batch_size]
            y_sliced = batch.y[:batch_size]

            # Validate labels before computing loss
            num_classes = out_sliced.shape[-1]
            if y_sliced.min() < 0 or y_sliced.max() >= num_classes:
                raise ValueError(
                    f"Invalid labels detected: min={y_sliced.min().item()}, "
                    f"max={y_sliced.max().item()}, but model has {num_classes} classes. "
                    f"Labels must be in range [0, {num_classes-1}]"
                )

            loss = self.criterion(out_sliced, y_sliced)
            loss.backward()
            self.optimiser.step()
            total_loss += float(loss.detach())

            pred = out_sliced.argmax(dim=1)

            all_preds.append(pred.cpu())
            all_labels.append(y_sliced.cpu())

            # Clean up batch to prevent memory accumulation
            del batch, out, out_sliced, y_sliced, loss, pred

        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()

        f1_illicit = f1_score(y_true, y_pred, pos_label=1, average='binary')  # illicit is class 1

        return total_loss / len(loader), f1_illicit

    def evaluate_loader(self, loader):
        self.model.eval()
        total_loss = 0
        all_preds, all_probs, all_labels = [], [], []
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = self.model(batch)

                batch_size = batch.batch_size
                out_sliced = out[:batch_size]
                y_sliced = batch.y[:batch_size]

                loss = self.criterion(out_sliced, y_sliced)
                total_loss += float(loss.detach())

                probs = torch.softmax(out_sliced, dim=1)
                pred = out_sliced.argmax(dim=1)

                all_preds.append(pred.cpu())
                all_probs.append(probs.cpu())
                all_labels.append(y_sliced.cpu())

                # Clean up batch to prevent memory accumulation
                del batch, out, out_sliced, y_sliced, loss, probs, pred

        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        y_prob = torch.cat(all_probs).numpy()

        metrics = calculate_metrics(y_true, y_pred, y_prob)

        metrics['probs'] = y_prob
        metrics['preds'] = y_pred
        metrics['y_true'] = y_true
        return total_loss / len(loader), metrics

    def train_step_full(self, data, mask):
        self.model.train()
        device = next(self.model.parameters()).device
        data = data.to(device)
        self.optimiser.zero_grad()
        train_mask = mask.to(device)
        out = self.model(data)

        out_sliced = out[train_mask]
        y_sliced = data.y[train_mask]

        loss = self.criterion(out_sliced, y_sliced)
        loss.backward()
        self.optimiser.step()

    def mini_eval_full(self, data, masks):
        self.model.eval()
        device = next(self.model.parameters()).device
        eval_mask = masks.to(device)
        with torch.no_grad():
            data = data.to(device)
            out = self.model(data)
            out_sliced = out[eval_mask]
            y_sliced = data.y[eval_mask]

            loss = self.criterion(out_sliced, y_sliced)

            pred = out_sliced.argmax(dim=1)

            f1_metric = BinaryF1Score().to(device)
            f1_illicit = f1_metric(pred, y_sliced).item()

            probs = torch.softmax(out_sliced, dim=1)[:, 1]
            pr_auc = _safe_average_precision(y_sliced.cpu().numpy(), probs.cpu().numpy())

        return float(loss.detach()), f1_illicit, pr_auc

    def evaluate_full(self, data, mask):
        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            data = data.to(device)
            val_mask = mask.to(device)
            out = self.model(data)

            out_sliced = out[val_mask]
            y_sliced = data.y[val_mask]

            loss = self.criterion(out_sliced, y_sliced)

            probs = torch.softmax(out_sliced, dim=1)
            pred = out_sliced.argmax(dim=1)

            y_true = y_sliced.cpu().numpy()
            y_pred = pred.cpu().numpy()
            y_prob = probs.cpu().numpy()

        metrics = calculate_metrics(y_true, y_pred, y_prob)

        metrics['probs'] = y_prob
        metrics['preds'] = y_pred
        metrics['y_true'] = y_true

        return float(loss.detach()), metrics

    def train_step_elliptic(self, data, mask):
        self.model.train()
        device = next(self.model.parameters()).device
        data = data.to(device)
        self.optimiser.zero_grad()

        out = self.model(data)

        # Extract performance masks and apply to out and y to compute loss and
        # metrics only on the known nodes of the training set.
        train_perf_eval_mask = mask[1].to(device)
        out_sliced = out[train_perf_eval_mask]
        y_sliced = data.y[train_perf_eval_mask]

        loss = self.criterion(out_sliced, y_sliced)
        loss.backward()
        self.optimiser.step()

    def evaluate_elliptic(self, data, masks):
        self.model.eval()
        device = next(self.model.parameters()).device
        # eval_mask is the full mask (used so all known nodes can be predicted);
        # performance is measured on the perf_eval_mask subset.
        eval_mask = masks[0].to(device)
        with torch.no_grad():
            data = data.to(device)
            out = self.model(data)

            perf_eval_mask = masks[1].to(device)
            out_sliced = out[perf_eval_mask]
            y_sliced = data.y[perf_eval_mask]

            loss = self.criterion(out_sliced, y_sliced)

            probs = torch.softmax(out_sliced, dim=1)
            pred = out_sliced.argmax(dim=1)

            y_true = y_sliced.cpu().numpy()
            y_pred = pred.cpu().numpy()
            y_prob = probs.cpu().numpy()

        metrics = calculate_metrics(y_true, y_pred, y_prob)

        metrics['probs'] = y_prob
        metrics['preds'] = y_pred
        metrics['y_true'] = y_true

        return float(loss.detach()), metrics

    def mini_eval_elliptic(self, data, masks):
        self.model.eval()
        device = next(self.model.parameters()).device
        eval_mask = masks[0].to(device)
        with torch.no_grad():
            data = data.to(device)
            out = self.model(data)

            perf_eval_mask = masks[1].to(device)
            out_sliced = out[perf_eval_mask]
            y_sliced = data.y[perf_eval_mask]

            loss = self.criterion(out_sliced, y_sliced)

            pred = out_sliced.argmax(dim=1)

            y_true_np = y_sliced.cpu().numpy()
            pred_np = pred.cpu().numpy()
            probs_np = torch.softmax(out_sliced, dim=1)[:, 1].cpu().numpy()

            f1_illicit = f1_score(
                y_true_np, pred_np, pos_label=1, average='binary', zero_division=0,
            )
            pr_auc = _safe_average_precision(y_true_np, probs_np)

        return float(loss.detach()), float(f1_illicit), float(pr_auc)

    # In-VRAM training path for graph-agnostic models (MLP). The full feature
    # and label tensors are pre-moved to GPU once by the caller; we just iterate
    # via index slicing. This bypasses NeighborLoader entirely, which otherwise
    # spends most of its time doing CPU graph sampling that the MLP throws away.
    def train_step_in_vram(self, x_train, y_train, batch_size):
        self.model.train()
        device = next(self.model.parameters()).device
        n = x_train.shape[0]
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        num_batches = 0
        all_preds, all_labels = [], []

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            # BatchNorm1d requires >=2 samples in train mode; skip the trailing
            # fragment when n isn't divisible by batch_size and the remainder is 1.
            if idx.shape[0] < 2:
                continue
            xb = x_train[idx]
            yb = y_train[idx]

            self.optimiser.zero_grad()
            out = self.model(SimpleNamespace(x=xb))

            loss = self.criterion(out, yb)
            loss.backward()
            self.optimiser.step()
            total_loss += float(loss.detach())
            num_batches += 1

            pred = out.argmax(dim=1)
            all_preds.append(pred.detach())
            all_labels.append(yb.detach())

        y_true = torch.cat(all_labels).cpu().numpy()
        y_pred = torch.cat(all_preds).cpu().numpy()
        f1_illicit = f1_score(y_true, y_pred, pos_label=1, average='binary')

        return total_loss / max(num_batches, 1), f1_illicit

    def mini_eval_in_vram(self, x_eval, y_eval, batch_size):
        self.model.eval()
        n = x_eval.shape[0]
        total_loss = 0.0
        num_batches = 0
        all_preds, all_probs, all_labels = [], [], []

        with torch.no_grad():
            for start in range(0, n, batch_size):
                xb = x_eval[start:start + batch_size]
                yb = y_eval[start:start + batch_size]
                out = self.model(SimpleNamespace(x=xb))

                loss = self.criterion(out, yb)
                total_loss += float(loss.detach())
                num_batches += 1

                pred = out.argmax(dim=1)
                probs = torch.softmax(out, dim=1)[:, 1]
                all_preds.append(pred)
                all_probs.append(probs)
                all_labels.append(yb)

        y_true = torch.cat(all_labels).cpu().numpy()
        y_pred = torch.cat(all_preds).cpu().numpy()
        y_prob = torch.cat(all_probs).cpu().numpy()
        f1_illicit = f1_score(y_true, y_pred, pos_label=1, average='binary')
        pr_auc = _safe_average_precision(y_true, y_prob)

        return total_loss / max(num_batches, 1), f1_illicit, pr_auc

    def evaluate_in_vram(self, x_eval, y_eval, batch_size):
        self.model.eval()
        n = x_eval.shape[0]
        total_loss = 0.0
        num_batches = 0
        all_preds, all_probs, all_labels = [], [], []

        with torch.no_grad():
            for start in range(0, n, batch_size):
                xb = x_eval[start:start + batch_size]
                yb = y_eval[start:start + batch_size]
                out = self.model(SimpleNamespace(x=xb))

                loss = self.criterion(out, yb)
                total_loss += float(loss.detach())
                num_batches += 1

                probs = torch.softmax(out, dim=1)
                pred = out.argmax(dim=1)
                all_preds.append(pred)
                all_probs.append(probs)
                all_labels.append(yb)

        y_true = torch.cat(all_labels).cpu().numpy()
        y_pred = torch.cat(all_preds).cpu().numpy()
        y_prob = torch.cat(all_probs).cpu().numpy()

        metrics = calculate_metrics(y_true, y_pred, y_prob)
        metrics['probs'] = y_prob
        metrics['preds'] = y_pred
        metrics['y_true'] = y_true
        return total_loss / max(num_batches, 1), metrics

    def evaluate_loader_mini(self, loader):
        self.model.eval()
        device = next(self.model.parameters()).device
        all_preds, all_probs, all_labels = [], [], []
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = self.model(batch)
                batch_size = batch.batch_size
                out_sliced = out[:batch_size]
                y_sliced = batch.y[:batch_size]

                loss = self.criterion(out_sliced, y_sliced)
                total_loss += float(loss.detach())

                pred = out_sliced.argmax(dim=1)
                probs = torch.softmax(out_sliced, dim=1)[:, 1]
                all_preds.append(pred.cpu())
                all_probs.append(probs.cpu())
                all_labels.append(y_sliced.cpu())

                del batch, out, out_sliced, y_sliced, loss, pred, probs

        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        y_prob = torch.cat(all_probs).numpy()
        f1_illicit = f1_score(y_true, y_pred, pos_label=1, average='binary')
        pr_auc = _safe_average_precision(y_true, y_prob)

        return total_loss / len(loader), f1_illicit, pr_auc


# %% GCN
class GCN(torch.nn.Module):
    """
    Graph Convolutional Network with tunable depth (n_layers in [1, 3]).
    n_layers=1 collapses to a single GCNConv(in, num_classes).
    """
    def __init__(self, num_node_features, num_classes, hidden_units, dropout=0.5, n_layers=2):
        super(GCN, self).__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_layers = n_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if n_layers == 1:
            self.convs.append(GCNConv(num_node_features, num_classes))
        else:
            self.convs.append(GCNConv(num_node_features, hidden_units))
            self.norms.append(GraphNorm(hidden_units))
            for _ in range(n_layers - 2):
                self.convs.append(GCNConv(hidden_units, hidden_units))
                self.norms.append(GraphNorm(hidden_units))
            self.convs.append(GCNConv(hidden_units, num_classes))

    def forward(self, data):
        x = data.x
        edge_index = data.adj_t if hasattr(data, 'adj_t') else data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.norms):
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# %% GAT
class GAT(nn.Module):
    """
    Graph Attention Network (GATv2) with tunable depth (n_layers in [1, 3]).
    n_layers=1 collapses to a single GATv2Conv(in, num_classes, heads=1, concat=False).
    """
    def __init__(self, num_node_features, num_classes, hidden_units, num_heads,
                 dropout_1=0.6, dropout_2=0.5, feature_dropout=0.5, n_layers=2):
        super(GAT, self).__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_layers = n_layers
        self.feature_dropout = feature_dropout

        per_head_dim = max(1, math.ceil(hidden_units / num_heads))
        total_hidden = per_head_dim * num_heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if n_layers == 1:
            self.convs.append(
                GATv2Conv(num_node_features, num_classes,
                          heads=1, concat=False, dropout=dropout_2)
            )
        else:
            self.convs.append(
                GATv2Conv(num_node_features, per_head_dim,
                          heads=num_heads, dropout=dropout_1)
            )
            self.norms.append(GraphNorm(total_hidden))
            for _ in range(n_layers - 2):
                self.convs.append(
                    GATv2Conv(total_hidden, per_head_dim,
                              heads=num_heads, dropout=dropout_1)
                )
                self.norms.append(GraphNorm(total_hidden))
            self.convs.append(
                GATv2Conv(total_hidden, num_classes,
                          heads=1, concat=False, dropout=dropout_2)
            )

    def forward(self, data):
        x = data.x
        edge_index = data.adj_t if hasattr(data, 'adj_t') else data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.norms):
                x = self.norms[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.feature_dropout, training=self.training)
        return x


# %% GIN
class GIN(nn.Module):
    """
    Graph Isomorphism Network with tunable depth (n_layers in [1, 3]).
    BatchNorm1d is kept inside the per-layer MLPs (standard GIN convention);
    a GraphNorm is applied to the final embedding before the linear head.
    """
    def __init__(self, num_node_features, num_classes, hidden_units, dropout=0.0, n_layers=2):
        super(GIN, self).__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_layers = n_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        for layer_idx in range(n_layers):
            in_dim = num_node_features if layer_idx == 0 else hidden_units
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_units),
                nn.BatchNorm1d(hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units),
            )
            self.convs.append(GINConv(mlp, train_eps=True))

        self.norm = GraphNorm(hidden_units)
        self.fc = nn.Linear(hidden_units, num_classes)

    def forward(self, data):
        x = data.x
        edge_index = data.adj_t if hasattr(data, 'adj_t') else data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.norm(x)
        x = F.relu(x)
        x = self.fc(x)
        return x


# %% MLP
class MLP(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_units, dropout_1=0.6, dropout_2=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_node_features, hidden_units)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.bn2 = nn.BatchNorm1d(hidden_units)
        self.fc3 = nn.Linear(hidden_units, num_classes)
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2

    def forward(self, data):
        x = data.x  # only use node features, no graph structure
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=self.dropout_1, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=self.dropout_2, training=self.training)
        x = self.fc3(x)
        return x
