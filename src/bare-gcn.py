# gcn_graph_classifier.py
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ----- normalization (same as before) -----
def normalize_adj_weighted(adj: torch.Tensor, self_loop_weight: float = 1.0,
                           mode: str = "symmetric", alpha: float = 0.1,
                           symmetrize: bool = True) -> torch.Tensor:
    N = adj.size(0)
    device = adj.device
    if mode == "symmetric":
        A = 0.5 * (adj + adj.T) if symmetrize else adj
        A_tilde = A + self_loop_weight * torch.eye(N, device=device)
        deg = A_tilde.sum(dim=1)
        deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        return D_inv_sqrt @ A_tilde @ D_inv_sqrt
    elif mode == "row":
        row_sum = adj.sum(dim=1, keepdim=True).clamp_min(1e-12)
        A_row = adj / row_sum
        I = torch.eye(N, device=device)
        return (1.0 - alpha) * A_row + alpha * I
    else:
        raise ValueError("mode must be 'symmetric' or 'row'")

# ----- barebones GCN -----
class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        out = A_hat @ (X @ self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out

class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.5, input_dropout: float = 0.1,
                 use_batchnorm: bool = True):
        super().__init__()
        self.g1 = GCNLayer(in_dim, hidden_dim)
        self.g2 = GCNLayer(hidden_dim, out_dim)
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity()
    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        h = F.dropout(X, p=self.input_dropout, training=self.training)
        h = self.g1(h, A_hat)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.g2(h, A_hat)   # [N, out_dim]
        return h

# ----- Graph-level classifier wrapper -----
class GraphLevelGCN(nn.Module):
    def __init__(self, in_dim: int = 128, hidden_dim: int = 32, node_out_dim: int = 64,
                 num_classes: int = 2, gcn_dropout: float = 0.5, gcn_input_dropout: float = 0.1,
                 use_batchnorm: bool = True, graph_dropout: float = 0.5):
        super().__init__()
        self.gcn = GCN(in_dim, hidden_dim, node_out_dim,
                       dropout=gcn_dropout, input_dropout=gcn_input_dropout,
                       use_batchnorm=use_batchnorm)
        self.graph_dropout = graph_dropout
        self.classifier = nn.Linear(node_out_dim, num_classes)
        self.ln = nn.LayerNorm(node_out_dim)
    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        """
        X: [N, 128], A_hat: [N, N]
        Returns logits for the graph: [num_classes]
        """
        node_emb = self.gcn(X, A_hat)         # [N, node_out_dim]
        graph_emb = node_emb.mean(dim=0)      # global mean pooling (replace with max/sum if desired)
        graph_emb = self.ln(graph_emb)
        graph_emb = F.normalize(graph_emb, p=2, dim=0)
        graph_emb = F.dropout(graph_emb, p=self.graph_dropout, training=self.training)
        logits = self.classifier(graph_emb)    # [C]
        return logits

# ----- regularization: DropEdge -----
def drop_edges(adj: torch.Tensor, drop_prob: float = 0.15, symmetrize: bool = True) -> torch.Tensor:
    """Randomly drop edges with probability drop_prob. Keeps weights of retained edges.
    Applies symmetric masking if requested.
    """
    if drop_prob <= 0.0:
        return adj
    device = adj.device
    mask = (torch.rand_like(adj, device=device) > drop_prob).float()
    if symmetrize:
        mask = ((mask + mask.T) >= 1.0).float()
    return adj * mask

if __name__ == "__main__":
    # -------- optional: offline data augmentation & caching --------
    # Creates augmented copies of each subject and caches them under
    # "src/gcn_input_augmented/" so we can train on a larger dataset.
    AUGMENT_DATA = False
    AUGMENTED_DIR = "src/gcn_input_augmented"
    INPUT_DIR = "src/gcn_input"
    NUM_AUGS_PER_SUBJECT = 2           # how many synthetic variants to create per subject
    FEATURE_JITTER_STD = 0.03          # gaussian noise std added to node features
    FEATURE_MASK_PROB = 0.02           # probability to zero-out individual feature entries
    EDGE_DROP_PROB_AUG = 0.03          # probability to drop edges in augmentation (separate from training-time DropEdge)
    EDGE_NOISE_STD = 0.01              # small gaussian noise added to retained edge weights (clipped to [0,1])
    RNG_SEED = 42

    def _augment_np_arrays(X_np_in: np.ndarray,
                           A_np_in: np.ndarray,
                           y_np_in: np.ndarray,
                           num_augs: int,
                           feat_jitter_std: float,
                           feat_mask_p: float,
                           edge_drop_p: float,
                           edge_noise_std: float,
                           seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rs = np.random.RandomState(seed)
        S, N, F = X_np_in.shape
        out_S = S * (1 + num_augs)
        X_out = np.empty((out_S, N, F), dtype=np.float32)
        A_out = np.empty((out_S, N, N), dtype=np.float32)
        y_out = np.empty((out_S,), dtype=y_np_in.dtype)

        write_idx = 0
        for s in range(S):
            X0 = X_np_in[s]
            A0 = A_np_in[s]
            y0 = y_np_in[s]

            # original
            X_out[write_idx] = X0
            A_out[write_idx] = A0
            y_out[write_idx] = y0
            write_idx += 1

            # augmented variants
            for _ in range(num_augs):
                # feature jitter
                noise = rs.normal(loc=0.0, scale=feat_jitter_std, size=X0.shape).astype(np.float32)
                X_aug = X0 + noise

                # feature masking
                if feat_mask_p > 0.0:
                    mask = (rs.rand(*X_aug.shape) > feat_mask_p).astype(np.float32)
                    X_aug = X_aug * mask

                # edge perturbation: drop and add small noise to retained weights
                if edge_drop_p > 0.0:
                    keep_mask = (rs.rand(*A0.shape) > edge_drop_p).astype(np.float32)
                    # symmetrize mask
                    keep_mask = ((keep_mask + keep_mask.T) >= 1.0).astype(np.float32)
                    A_aug = A0 * keep_mask
                else:
                    A_aug = A0.copy()
                if edge_noise_std > 0.0:
                    e_noise = rs.normal(loc=0.0, scale=edge_noise_std, size=A_aug.shape).astype(np.float32)
                    A_aug = A_aug + e_noise
                # ensure valid range and symmetry
                A_aug = np.clip(0.5 * (A_aug + A_aug.T), 0.0, 1.0)

                X_out[write_idx] = X_aug.astype(np.float32)
                A_out[write_idx] = A_aug.astype(np.float32)
                y_out[write_idx] = y0
                write_idx += 1

        return X_out, A_out, y_out

    def _ensure_augmented_cache():
        os.makedirs(AUGMENTED_DIR, exist_ok=True)
        out_X = os.path.join(AUGMENTED_DIR, "subject_embedding_matrices.npy")
        out_A = os.path.join(AUGMENTED_DIR, "subject_adjacency_matrices.npy")
        out_y = os.path.join(AUGMENTED_DIR, "labels.npy")
        if os.path.exists(out_X) and os.path.exists(out_A) and os.path.exists(out_y):
            # verify group size matches current config (1 + NUM_AUGS_PER_SUBJECT)
            try:
                in_y_path = os.path.join(INPUT_DIR, "labels.npy")
                if os.path.exists(in_y_path):
                    y_in_len = int(np.load(in_y_path).shape[0])
                    y_out_len = int(np.load(out_y).shape[0])
                    if y_in_len > 0 and y_out_len % y_in_len == 0:
                        current_group_size = y_out_len // y_in_len
                        desired_group_size = 1 + NUM_AUGS_PER_SUBJECT
                        if current_group_size == desired_group_size:
                            return
                        # else: fall through to regenerate
                    else:
                        # cannot infer; regenerate
                        pass
                else:
                    # cannot validate without originals; keep existing cache
                    return
            except Exception:
                # On any failure, keep the existing cache
                return
        in_X = os.path.join(INPUT_DIR, "subject_embedding_matrices.npy")
        in_A = os.path.join(INPUT_DIR, "subject_adjacency_matrices.npy")
        in_y = os.path.join(INPUT_DIR, "labels.npy")
        X_src = np.load(in_X)
        A_src = np.load(in_A)
        y_src = np.load(in_y)
        X_aug, A_aug, y_aug = _augment_np_arrays(
            X_src, A_src, y_src,
            num_augs=NUM_AUGS_PER_SUBJECT,
            feat_jitter_std=FEATURE_JITTER_STD,
            feat_mask_p=FEATURE_MASK_PROB,
            edge_drop_p=EDGE_DROP_PROB_AUG,
            edge_noise_std=EDGE_NOISE_STD,
            seed=RNG_SEED,
        )
        np.save(out_X, X_aug)
        np.save(out_A, A_aug)
        np.save(out_y, y_aug)

    if AUGMENT_DATA:
        _ensure_augmented_cache()
        DATA_DIR = AUGMENTED_DIR
    else:
        DATA_DIR = INPUT_DIR

    # -------- load your data --------
    X_np = np.load(os.path.join(DATA_DIR, "subject_embedding_matrices_nofinetune.npy"))   # (S, N, 128)
    A_np = np.load(os.path.join(DATA_DIR, "subject_adjacency_matrices.npy"))   # (S, N, N), values in [0,1]
    y_np = np.load(os.path.join(DATA_DIR, "labels.npy"))                       # (S,)

    S, N, F_IN = X_np.shape[0], X_np.shape[1], X_np.shape[2]
    assert F_IN == 128, f"Expected embeddings of length 128, got {F_IN}"
    assert A_np.shape == (S, N, N), "A must be (S, N, N)"
    assert y_np.shape == (S,), "y must be (S,) for graph labels"

    # Torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Reproducibility settings
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    X_all = torch.from_numpy(X_np).float().to(device)   # (S, N, 128)
    A_all = torch.from_numpy(A_np).float().to(device)   # (S, N, N)
    y_all = torch.from_numpy(y_np).long().to(device)    # (S,)

    # Optional: L2-normalize node embeddings within each subject
    X_all = F.normalize(X_all, p=2, dim=2)

    num_classes = int(y_all.max().item() + 1)
    model = GraphLevelGCN(
        in_dim=128, hidden_dim=32, node_out_dim=64, num_classes=num_classes,
        gcn_dropout=0.2, gcn_input_dropout=0.05, use_batchnorm=True, graph_dropout=0.2
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    # class weights for imbalance
    with torch.no_grad():
        class_counts = torch.bincount(torch.from_numpy(y_np).long(), minlength=num_classes).float()
        class_weights = (class_counts.sum() / (class_counts + 1e-8)).to(device)
    try:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    except TypeError:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Stratified split by groups (original subject + its augmentations as one group),
    # and keep validation/test sets un-augmented (only first/original item from each group).
    def grouped_stratified_split_indices(y_tensor: torch.Tensor,
                                         group_size: int,
                                         train_ratio: float,
                                         val_ratio: float,
                                         seed: int = 42):
        assert y_tensor.numel() % group_size == 0, "y length must be divisible by group_size"
        assert 0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0 and train_ratio + val_ratio < 1.0, "invalid ratios"
        test_ratio = 1.0 - train_ratio - val_ratio
        generator = torch.Generator()
        generator.manual_seed(seed)
        num_groups = y_tensor.numel() // group_size
        # label per group from the first element in each group
        y_groups = y_tensor.view(num_groups, group_size)[:, 0].detach().cpu()
        classes = torch.unique(y_groups)
        train_idx_parts = []
        val_idx_parts = []
        test_idx_parts = []
        for c in classes.tolist():
            group_ids_c = (y_groups == c).nonzero(as_tuple=True)[0]
            if group_ids_c.numel() == 0:
                continue
            perm = torch.randperm(group_ids_c.numel(), generator=generator)
            group_ids_c = group_ids_c[perm]
            n_c = group_ids_c.numel()
            n_train = int(train_ratio * n_c)
            n_val = int(val_ratio * n_c)
            # ensure we don't exceed bounds
            n_train = min(n_train, n_c)
            n_val = min(n_val, max(n_c - n_train, 0))
            split_train = n_train
            split_val = n_train + n_val
            train_groups = group_ids_c[:split_train]
            val_groups = group_ids_c[split_train:split_val]
            test_groups = group_ids_c[split_val:]
            # map groups to indices
            for g in train_groups.tolist():
                start = g * group_size
                end = start + group_size
                train_idx_parts.append(torch.arange(start, end, dtype=torch.long))
            for g in val_groups.tolist():
                start = g * group_size
                # only take the original (first) item in the group for validation
                val_idx_parts.append(torch.tensor([start], dtype=torch.long))
            for g in test_groups.tolist():
                start = g * group_size
                # only take the original (first) item in the group for test
                test_idx_parts.append(torch.tensor([start], dtype=torch.long))
        train_idx = torch.cat(train_idx_parts) if len(train_idx_parts) > 0 else torch.empty(0, dtype=torch.long)
        val_idx = torch.cat(val_idx_parts) if len(val_idx_parts) > 0 else torch.empty(0, dtype=torch.long)
        test_idx = torch.cat(test_idx_parts) if len(test_idx_parts) > 0 else torch.empty(0, dtype=torch.long)
        # Shuffle
        if train_idx.numel() > 0:
            train_idx = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
        if val_idx.numel() > 0:
            val_idx = val_idx[torch.randperm(val_idx.numel(), generator=generator)]
        if test_idx.numel() > 0:
            test_idx = test_idx[torch.randperm(test_idx.numel(), generator=generator)]
        return train_idx, val_idx, test_idx

    # Infer GROUP_SIZE from file lengths to avoid cache/config mismatches
    if AUGMENT_DATA:
        try:
            orig_y_path = os.path.join(INPUT_DIR, "labels.npy")
            if os.path.exists(orig_y_path):
                orig_S = int(np.load(orig_y_path).shape[0])
                cur_S = int(y_np.shape[0])
                GROUP_SIZE = cur_S // orig_S if orig_S > 0 and cur_S % orig_S == 0 else 1
            else:
                GROUP_SIZE = 1
        except Exception:
            GROUP_SIZE = 1
    else:
        GROUP_SIZE = 1

    # Split ratios
    VAL_RATIO = 0.10
    TEST_RATIO = 0.10
    TRAIN_RATIO = 1.0 - VAL_RATIO - TEST_RATIO

    if GROUP_SIZE > 1:
        train_idx, val_idx, test_idx = grouped_stratified_split_indices(
            y_all, group_size=GROUP_SIZE, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=42
        )
    else:
        # fallback to simple stratified split when no augmentation
        def stratified_split_indices_three(y_tensor: torch.Tensor, train_ratio: float, val_ratio: float, seed: int = 42):
            assert 0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0 and train_ratio + val_ratio < 1.0, "invalid ratios"
            generator = torch.Generator(); generator.manual_seed(seed)
            y_cpu = y_tensor.detach().cpu(); classes = torch.unique(y_cpu)
            train_parts = []; val_parts = []; test_parts = []
            for c in classes.tolist():
                idx_c = (y_cpu == c).nonzero(as_tuple=True)[0]
                if idx_c.numel() == 0:
                    continue
                perm = torch.randperm(idx_c.numel(), generator=generator)
                idx_c = idx_c[perm]
                n_c = idx_c.numel()
                n_train = int(train_ratio * n_c)
                n_val = int(val_ratio * n_c)
                n_train = min(n_train, n_c)
                n_val = min(n_val, max(n_c - n_train, 0))
                train_parts.append(idx_c[:n_train])
                val_parts.append(idx_c[n_train:n_train+n_val])
                test_parts.append(idx_c[n_train+n_val:])
            train_idx = torch.cat(train_parts) if len(train_parts) > 0 else torch.empty(0, dtype=torch.long)
            val_idx = torch.cat(val_parts) if len(val_parts) > 0 else torch.empty(0, dtype=torch.long)
            test_idx = torch.cat(test_parts) if len(test_parts) > 0 else torch.empty(0, dtype=torch.long)
            if train_idx.numel() > 0:
                train_idx = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
            if val_idx.numel() > 0:
                val_idx = val_idx[torch.randperm(val_idx.numel(), generator=generator)]
            if test_idx.numel() > 0:
                test_idx = test_idx[torch.randperm(test_idx.numel(), generator=generator)]
            return train_idx, val_idx, test_idx
        train_idx, val_idx, test_idx = stratified_split_indices_three(y_all, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=42)

    EDGE_DROP_PROB = 0.10

    # ----- early stopping and metric tracking -----
    EARLY_STOP_PATIENCE = 10
    best_test_acc = -1.0
    best_epoch = -1
    epochs_since_improve = 0
    best_state = None
    use_early_stop = test_idx.numel() > 0

    history = {
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
    }

    def forward_subject(s: int):
        X = X_all[s]             # [N,128]
        A = A_all[s]             # [N,N]
        A_used = drop_edges(A, drop_prob=EDGE_DROP_PROB, symmetrize=True) if model.training else A
        A_hat = normalize_adj_weighted(A_used, self_loop_weight=1.0, mode="symmetric", symmetrize=True)
        return model(X, A_hat)   # [C]

    # Helpers for evaluation metrics
    def acc_on(indices: torch.Tensor) -> float:
        with torch.no_grad():
            correct = 0; total = 0
            for s in indices.tolist():
                logits = forward_subject(s)
                pred = logits.argmax().item()
                correct += int(pred == y_all[s].item())
                total += 1
            return correct / max(total, 1)

    def loss_on(indices: torch.Tensor) -> float:
        with torch.no_grad():
            total = 0; tot_loss = 0.0
            for s in indices.tolist():
                logits = forward_subject(s)
                tot_loss += criterion(logits.unsqueeze(0), y_all[s].unsqueeze(0)).item()
                total += 1
            return tot_loss / max(total, 1)

    # Train loop (per-subject forward; no batching for simplicity)
    for epoch in range(200):
        model.train()
        total_loss = 0.0
        for s in train_idx.tolist():
            opt.zero_grad()
            logits = forward_subject(s)
            loss = criterion(logits.unsqueeze(0), y_all[s].unsqueeze(0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            total_loss += loss.item()

        # Eval
        model.eval()
        train_acc = acc_on(train_idx)
        val_acc = acc_on(val_idx)
        val_loss = loss_on(val_idx)
        test_acc = acc_on(test_idx)
        test_loss = loss_on(test_idx)
        scheduler.step(val_loss)

        avg_train_loss = total_loss / max(len(train_idx), 1)

        # record metrics
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)

        # early stopping on test accuracy
        if use_early_stop:
            if test_acc > best_test_acc + 1e-12:
                best_test_acc = test_acc
                best_epoch = epoch
                epochs_since_improve = 0
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping at epoch {epoch+1:03d} (best test_acc={best_test_acc:.3f} at epoch {best_epoch+1:03d})")
                    break

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1:03d} | loss={avg_train_loss:.4f} "
                f"| train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | test_acc={test_acc:.3f} "
                f"| val_loss={val_loss:.4f} | test_loss={test_loss:.4f}"
            )


    # Final evaluation on validation and test sets
    model.eval()
    if best_state is not None:
        model.load_state_dict(best_state)
    final_train_acc = acc_on(train_idx)
    final_train_loss = loss_on(train_idx)
    final_val_acc = acc_on(val_idx)
    final_val_loss = loss_on(val_idx)
    test_acc = acc_on(test_idx)
    test_loss = loss_on(test_idx)
    print(
        f"Final | train_acc={final_train_acc:.3f} | train_loss={final_train_loss:.4f} "
        f"| val_acc={final_val_acc:.3f} | val_loss={final_val_loss:.4f} "
        f"| test_acc={test_acc:.3f} | test_loss={test_loss:.4f}"
    )

    # Inference per subject
    model.eval()
    with torch.no_grad():
        preds = []
        for s in range(S):
            logits = forward_subject(s)
            preds.append(int(logits.argmax().item()))
    print("Preds (per subject):", preds)

    # ----- visualization: stacked subplots for accuracy and loss -----
    try:
        epochs = list(range(1, len(history["train_acc"]) + 1))
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Accuracy subplot
        axes[0].plot(epochs, history["train_acc"], label="Train Acc")
        axes[0].plot(epochs, history["val_acc"], label="Val Acc")
        axes[0].plot(epochs, history["test_acc"], label="Test Acc")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)

        # Loss subplot
        axes[1].plot(epochs, history["train_loss"], label="Train Loss")
        axes[1].plot(epochs, history["val_loss"], label="Val Loss")
        axes[1].plot(epochs, history["test_loss"], label="Test Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), "training_metrics.png")
        plt.savefig(out_path, dpi=150)
        print(f"Saved training metrics plot to: {out_path}")
    except Exception as e:
        print(f"Failed to generate training metrics plot: {e}")
