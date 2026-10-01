import logging
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from utils import set_seed, load_data
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score,
    adjusted_rand_score, matthews_corrcoef,
    davies_bouldin_score, silhouette_score,
)

from psgcl_core.data import dataset_mat_path
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import argparse
import json
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def mat_dataset_name(dataset: str) -> str:
    return "BRCA" if dataset == "BRCA-v5" else dataset


def relabel(labels: np.ndarray):
    uniq = np.unique(labels)
    mapping = {u: i for i, u in enumerate(uniq)}
    y = np.array([mapping[v] for v in labels], dtype=np.int64)
    return y, mapping

def standardize_split(X_train: np.ndarray, X_test: np.ndarray):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def build_view_list(gene: np.ndarray, methylation: np.ndarray, mirna: np.ndarray):
    return [gene.astype(np.float32), methylation.astype(np.float32), mirna.astype(np.float32)]
VIEW_NAMES = ["mRNA_expression", "DNA_methylation", "miRNA_expression"]


def compute_covariance(X: np.ndarray, eps: float=1e-10) -> np.ndarray:
    N = X.shape[0]
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    cov = (Xc.T @ Xc) / max((N - 1), 1)
    cov += eps * np.eye(cov.shape[0], dtype=cov.dtype)
    return cov

def mahalanobis_distance_matrix(X: np.ndarray, cov: np.ndarray = None, eps: float = 1e-10) -> np.ndarray:
    n_samples = X.shape[0]
    centered = X - X.mean(axis=0, keepdims=True)
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values ** 2 / max(n_samples - 1, 1)
    factors = singular_values ** 2 / (eigenvalues + eps)
    gram = (left * factors) @ left.T
    diagonal = np.diag(gram)
    dist2 = diagonal[:, None] + diagonal[None, :] - 2.0 * gram
    np.maximum(dist2, 0.0, out=dist2)
    dist = np.sqrt(dist2)
    np.fill_diagonal(dist, 0.0)
    return dist


def compute_density(dist_matrix: np.ndarray, k: int) -> np.ndarray:
    N = dist_matrix.shape[0]
    k_int = int(max(1, min(round(k), N - 1)))
    order = np.argsort(dist_matrix, axis=1)
    knn_idx = order[:, 1:k_int + 1]
    density = np.take_along_axis(dist_matrix, knn_idx, axis=1).mean(axis=1)
    return density

def build_adjacency(dist_matrix: np.ndarray, density: np.ndarray, k: float) -> np.ndarray:
    N = dist_matrix.shape[0]
    threshold = density * k
    A = np.zeros((N, N), dtype=np.float32)
    mask = dist_matrix <= threshold[:, None]
    np.fill_diagonal(mask, False)
    A[mask] = 1.0 / (1.0 + dist_matrix[mask])
    A = np.maximum(A, A.T)
    return A

def normalize_laplacian(A: np.ndarray) -> np.ndarray:
    N = A.shape[0]
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    deg_inv_sqrt[nz] = np.power(deg[nz], -0.5)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    L = np.eye(N, dtype=np.float32) - (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)
    return L

def build_graph(X: np.ndarray, k: int, eps: float = 1e-10):
    dist = mahalanobis_distance_matrix(X, eps=eps)
    density = compute_density(dist, k)
    A = build_adjacency(dist, density, k)
    L = normalize_laplacian(A)
    return L.astype(np.float32), A.astype(np.float32)


def build_graph_euclidean(X: np.ndarray, k: int):
    dist = cdist(X, X, metric="euclidean").astype(np.float32)
    density = compute_density(dist, k)
    A = build_adjacency(dist, density, k)
    L = normalize_laplacian(A)
    return L.astype(np.float32), A.astype(np.float32)

class GraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        support = X @  self.weight
        out = L @ support
        if self.bias is not None:
            out += self.bias
        return out
    

class GCN_E(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(128, 256, 256), dropout: float=0.5, negative_slope: float=0.25):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        self.layers = nn.ModuleList(
            [GraphConvolution(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        H = X
        for layer in self.layers:
            H = layer(H, L)
            H = self.act(H)
            H = self.dropout(H)
        return H

class InterViewAttention(nn.Module):
    def __init__(self, num_views: int, feat_dim: int):
        super().__init__()
        self.num_views = num_views
        self.transform = nn.ModuleList(
            [nn.Linear(feat_dim, feat_dim, bias=False) for _ in range(num_views)]
        )
        self.attn_vec = nn.ParameterList(
            [nn.Parameter(torch.randn(feat_dim) * 0.01) for _ in range(num_views)]
        )
    
    
    def forward(self, H_list):
        V = self.num_views
        Hp = [self.transform[v](H_list[v]) for v in range(V)]
        scores = torch.stack(
            [(Hp[v] * self.attn_vec[v]).sum(dim=1) for v in range(V)], dim=1
        )
        alpha = F.softmax(scores, dim=1)
        Hatt = sum(alpha[:, v:v + 1] * H_list[v] for v in range(V))
        H_final = [H_list[v] + Hatt - alpha[:, v:v + 1] * Hp[v] for v in range(V)]
        return H_final, alpha
    
class VCDN(nn.Module):
    def __init__(self, num_views: int, num_classes: int, hidden_dim: int=256):
        super().__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        in_dim = num_classes ** num_views
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, prob_list):
        x = prob_list[0]
        for p in prob_list[1:]:
            x = torch.einsum("bi,bj->bij", x, p).reshape(x.size(0), -1)
        return self.net(x)
    
class MCgnnFull(nn.Module):
    def __init__(self, in_dims, num_classes, hidden_dims=(128, 256, 256), dropout=0.5, vcdn_hidden=256):
        super().__init__()
        self.num_views = len(in_dims)
        self.num_classes = num_classes
        feat_dim = hidden_dims[-1]
        self.encoders = nn.ModuleList(
            [GCN_E(in_dim, hidden_dims, dropout) for in_dim in in_dims]
        )
        self.attention = InterViewAttention(self.num_views, feat_dim)
        self.classifiers = nn.ModuleList(
            [nn.Linear(feat_dim, num_classes) for _ in range(self.num_views)]
        )
        self.vcdn = VCDN(self.num_views, num_classes, vcdn_hidden)

    def forward(self, X_list, L_list, return_embedding=False):
        H_list = [self.encoders[v](X_list[v], L_list[v]) for v in range(self.num_views)]
        H_final, alpha = self.attention(H_list)
        logits_list = [self.classifiers[v](H_final[v]) for v in range(self.num_views)]
        probs_list = [F.softmax(l, dim=1) for l in logits_list]
        sig_probs = [torch.sigmoid(p) for p in probs_list]
        vcdn_logits = self.vcdn(sig_probs)
        if return_embedding:
            embedding = torch.stack(H_final, dim=0).mean(dim=0)
            return logits_list, vcdn_logits, alpha, embedding
        return logits_list, vcdn_logits, alpha
    
    def param_groups(self, lr_encoder=5e-4, lr_vcdn=1e-3):
        enc_params = list(self.encoders.parameters()) + \
            list(self.attention.parameters()) + list(self.classifiers.parameters())
        return [
            {"params": enc_params, "lr": lr_encoder},
            {"params": self.vcdn.parameters(), "lr": lr_vcdn},
        ]
 
    def pretrain_params(self, lr=1e-3):
        enc_params = list(self.encoders.parameters()) + \
            list(self.attention.parameters()) + list(self.classifiers.parameters())
        return [{"params": enc_params, "lr": lr}]
    


METRIC_KEYS = ["precision", "recall", "f1", "f1_weighted", "acc",
               "auc", "prauc", "ari", "mcc", "dbi", "ss"]


def _safe_auc_prauc(y_true, y_prob, n_classes):
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
            prauc = average_precision_score(y_true, y_prob[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            present = y_bin.sum(axis=0) > 0
            auc = roc_auc_score(y_bin[:, present], y_prob[:, present], average="macro", multi_class="ovr")
            prauc = average_precision_score(y_bin[:, present], y_prob[:, present], average="macro")
    except ValueError:
        auc, prauc = float("nan"), float("nan")
    return auc, prauc


def _safe_cluster_metrics(embedding, cluster_labels):
    try:
        if len(np.unique(cluster_labels)) < 2 or embedding.shape[0] < 3:
            return float("nan"), float("nan")
        dbi = davies_bouldin_score(embedding, cluster_labels)
        ss = silhouette_score(embedding, cluster_labels)
    except Exception:
        dbi, ss = float("nan"), float("nan")
    return dbi, ss


def compute_all_metrics(y_true, y_pred, y_prob, embedding, n_classes):
    p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc, prauc = _safe_auc_prauc(y_true, y_prob, n_classes)
    dbi, ss = _safe_cluster_metrics(embedding, y_pred)

    return {
        "precision": p, "recall": r, "f1": f1, "f1_weighted": f1w, "acc": acc,
        "auc": auc, "prauc": prauc, "ari": ari, "mcc": mcc, "dbi": dbi, "ss": ss,
    }

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
def to_tensor(x, device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)
 
 
def compute_loss(logits_list, vcdn_logits, labels, ce=None):
    ce = ce or nn.CrossEntropyLoss()
    loss = sum(ce(logits, labels) for logits in logits_list)
    loss = loss + ce(vcdn_logits, labels)
    return loss


def train_one_fold(view_list, y, train_idx, test_idx, k=10, num_classes=None,
                    hidden_dims=(128, 256, 256), dropout=0.5, vcdn_hidden=256,
                    pretrain_epochs=100, main_epochs=200,
                    lr_pretrain=1e-3, lr_encoder=5e-4, lr_vcdn=1e-3, eps=1e-10,
                    device=None, verbose=False, checkpoint_path=None):
    device = device or get_device()
    num_classes = num_classes or int(y.max() + 1)
    n_views = len(view_list)
 
    graph_builder =build_graph
 
    Xtr_list, Ltr_list = [], []
    for v in range(n_views):
        Xtr, _, _ = standardize_split(view_list[v][train_idx], view_list[v][test_idx])
        Xtr_list.append(Xtr)
        L, _ = graph_builder(Xtr, k, eps)
        Ltr_list.append(L)
 
    Xtrte_list, Ltrte_list = [], []
    for v in range(n_views):
        Xtr, Xte, scaler = standardize_split(view_list[v][train_idx], view_list[v][test_idx])
        Xall = np.concatenate([Xtr, Xte], axis=0)
        Xtrte_list.append(Xall)
        L, _ = graph_builder(Xall, k, eps)
        Ltrte_list.append(L)
 
    n_train = len(train_idx)
    y_train = y[train_idx]
 
    Xtr_t = [to_tensor(x, device) for x in Xtr_list]
    Ltr_t = [to_tensor(l, device) for l in Ltr_list]
    ytr_t = to_tensor(y_train, device, dtype=torch.long)
 
    Xtrte_t = [to_tensor(x, device) for x in Xtrte_list]
    Ltrte_t = [to_tensor(l, device) for l in Ltrte_list]
 
    in_dims = [x.shape[1] for x in Xtr_list]
    model = MCgnnFull(in_dims, num_classes, hidden_dims, dropout, vcdn_hidden).to(device)
    ce = nn.CrossEntropyLoss()
 
    opt_pre = torch.optim.Adam(model.pretrain_params(lr=lr_pretrain))
    model.train()
    for epoch in range(pretrain_epochs):
        opt_pre.zero_grad()
        logits_list, _, _ = model(Xtr_t, Ltr_t)
        loss = sum(ce(logits, ytr_t) for logits in logits_list)
        loss.backward()
        opt_pre.step()
        if verbose and (epoch + 1) % 25 == 0:
            print(f"[pretrain] epoch {epoch+1}/{pretrain_epochs} loss={loss.item():.4f}")
 
    opt_main = torch.optim.Adam(model.param_groups(lr_encoder=lr_encoder, lr_vcdn=lr_vcdn))
    for epoch in range(main_epochs):
        model.train()
        opt_main.zero_grad()
        logits_list, vcdn_logits, _ = model(Xtr_t, Ltr_t)
        loss = compute_loss(logits_list, vcdn_logits, ytr_t, ce)
        loss.backward()
        opt_main.step()
        if verbose and (epoch + 1) % 50 == 0:
            print(f"[main] epoch {epoch+1}/{main_epochs} loss={loss.item():.4f}")
 
    model.eval()
    with torch.no_grad():
        _, vcdn_logits_all, _, embedding_all = model(Xtrte_t, Ltrte_t, return_embedding=True)
        prob_all = torch.softmax(vcdn_logits_all, dim=1).cpu().numpy()
        preds_all = prob_all.argmax(axis=1)
        embedding_all = embedding_all.cpu().numpy()
 
    y_pred_train, y_pred_test = preds_all[:n_train], preds_all[n_train:]
    prob_train, prob_test = prob_all[:n_train], prob_all[n_train:]
    emb_train, emb_test = embedding_all[:n_train], embedding_all[n_train:]
 
    metrics_test = compute_all_metrics(y[test_idx], y_pred_test, prob_test, emb_test, num_classes)
    metrics_train = compute_all_metrics(y_train, y_pred_train, prob_train, emb_train, num_classes)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        torch.save(
            {
                "format_version": 1,
                "in_dims": in_dims,
                "num_classes": num_classes,
                "hidden_dims": tuple(hidden_dims),
                "dropout": dropout,
                "vcdn_hidden": vcdn_hidden,
                "k": k,
                "model_state": model.state_dict(),
                "train_indices": np.asarray(train_idx).tolist(),
                "test_indices": np.asarray(test_idx).tolist(),
            },
            checkpoint_path,
        )
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        reloaded = MCgnnFull(
            payload["in_dims"], payload["num_classes"], payload["hidden_dims"],
            payload["dropout"], payload["vcdn_hidden"],
        ).to(device)
        reloaded.load_state_dict(payload["model_state"])
        reloaded.eval()
        with torch.no_grad():
            _, reloaded_logits, _, _ = reloaded(Xtrte_t, Ltrte_t, return_embedding=True)
            reloaded_prob = torch.softmax(reloaded_logits, dim=1).cpu().numpy()
        if not np.allclose(reloaded_prob, prob_all, rtol=1e-5, atol=1e-6):
            raise RuntimeError("Reloaded MCGNN checkpoint changed inference probabilities.")
 
    extra_test = {"y_true": y[test_idx], "y_pred": y_pred_test, "y_prob": prob_test, "embedding": emb_test}
    return metrics_test, metrics_train, model, extra_test

_HEADER = "Precision | Recall | F1 Score | F1 Score (Weighted) | ACC | AUC | PR AUC | ARI | MCC | DBI | SS"
_ROW_ORDER = ["precision", "recall", "f1", "f1_weighted", "acc", "auc", "prauc", "ari", "mcc", "dbi", "ss"]


def _format_metric(value):
    return "NaN" if not np.isfinite(value) else f"{value:.6f}"


def _format_fold_report(seed, fold, total_folds, metrics):
    values = "  ".join(f"{key}={_format_metric(metrics[key])}" for key in _ROW_ORDER)
    return f"Completed seed={seed}, fold={fold}/{total_folds}\n  {values}"


def _format_summary_report(title, scores):
    lines = [f"{title} (mean +/- std):"]
    for key in _ROW_ORDER:
        mean, std = _mean_std(scores, key)
        lines.append(f"  {key}={_format_metric(mean)} +/- {_format_metric(std)}")
    return "\n".join(lines)


def _append_log(path, report):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(report)
        handle.write("\n\n")
 
 
def _write_fold_row(f, m):
    f.write(
        "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" % tuple(m[k] for k in _ROW_ORDER)
    )
 
 
def _mean_std(scores, key):
    arr = np.array(scores[key], dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std())


def _seed_mean_scores(per_seed_scores):
    return {
        key: [_mean_std(scores, key)[0] for scores in per_seed_scores.values()]
        for key in METRIC_KEYS
    }
 
 
def _write_summary(f, scores, header_line=None):
    p_m, p_s = _mean_std(scores, "precision")
    r_m, r_s = _mean_std(scores, "recall")
    f1_m, f1_s = _mean_std(scores, "f1")
    f1w_m, f1w_s = _mean_std(scores, "f1_weighted")
    acc_m, acc_s = _mean_std(scores, "acc")
    ari_m, ari_s = _mean_std(scores, "ari")
    mcc_m, mcc_s = _mean_std(scores, "mcc")
    auc_m, auc_s = _mean_std(scores, "auc")
    prauc_m, prauc_s = _mean_std(scores, "prauc")
    dbi_m, dbi_s = _mean_std(scores, "dbi")
    ss_m, ss_s = _mean_std(scores, "ss")
 
    if header_line:
        f.write(header_line)
 
    f.write(
        "Precision: %.4f ± %.4f | "
        "Recall: %.4f ± %.4f | "
        "F1 Score: %.4f ± %.4f | "
        "F1 Score (Weighted): %.4f ± %.4f | "
        "ACC: %.4f ± %.4f | "
        "ARI: %.4f ± %.4f | "
        "MCC: %.4f ± %.4f | "
        "AUC: %.4f ± %.4f | "
        "PR AUC: %.4f ± %.4f | "
        "DBI: %.4f ± %.4f | "
        "SS: %.4f ± %.4f\n"
        % (
            p_m, p_s, r_m, r_s, f1_m, f1_s, f1w_m, f1w_s, acc_m, acc_s,
            ari_m, ari_s, mcc_m, mcc_s, auc_m, auc_s, prauc_m, prauc_s,
            dbi_m, dbi_s, ss_m, ss_s,
        )
    )
 
 
def run_experiment(view_list, y, result_path, name_file,
                    seeds=(223, 777, 2026), n_splits=10, k=10,
                    overall_log_name=None, device=None, verbose_fold=True, checkpoint_dir=None, **train_kwargs):

    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    seed_log_dir = result_path / "seed_logs"
    seed_log_dir.mkdir(exist_ok=True)
    log_path = result_path / f"{name_file}.log"
    overall_path = result_path / f"{overall_log_name}.log" if overall_log_name else log_path
    device = device or get_device()
 
    all_scores = {key: [] for key in METRIC_KEYS}
    per_seed_scores = {}
 
    _append_log(log_path, "MCGNN training started")

    for seed in seeds:
        set_seed(seed)
        seed_log_path = seed_log_dir / f"seed_{seed}.log"
        _append_log(seed_log_path, f"Results for seed={seed}")
 
        seed_scores = {key: [] for key in METRIC_KEYS}
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
 
        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            metrics_test, _, _, _ = train_one_fold(
                view_list, y, train_idx, test_idx, k=k, device=device,
                checkpoint_path=(Path(checkpoint_dir) / f"seed_{seed}_fold_{fold + 1}.pt") if checkpoint_dir else None,
                **train_kwargs
            )
 
            for key in METRIC_KEYS:
                seed_scores[key].append(metrics_test[key])
                all_scores[key].append(metrics_test[key])
 
            fold_report = _format_fold_report(seed, fold + 1, n_splits, metrics_test)
            _append_log(log_path, fold_report)
            _append_log(seed_log_path, fold_report)

            if verbose_fold:
                print(fold_report, flush=True)

        seed_summary = _format_summary_report(f"Seed {seed} summary", seed_scores)
        _append_log(log_path, seed_summary)
        _append_log(seed_log_path, seed_summary)
        if verbose_fold:
            print(seed_summary, flush=True)

        per_seed_scores[seed] = seed_scores

    final_summary = _format_summary_report(
        f"Final summary across {len(per_seed_scores)} seed means",
        _seed_mean_scores(per_seed_scores),
    )
    _append_log(overall_path, final_summary)
    if verbose_fold:
        print(final_summary, flush=True)
 
    return all_scores, per_seed_scores


def evaluate_checkpoint(checkpoint_path, data_path, dataset, device=None):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}.")
    device = device or get_device()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gene, methylation, mirna, labels, _ = load_data(data_path, mat_dataset_name(dataset))
    y, _ = relabel(labels)
    views = build_view_list(gene, methylation, mirna)
    tr_idx = np.asarray(payload["train_indices"], dtype=np.int64)
    te_idx = np.asarray(payload["test_indices"], dtype=np.int64)
    x_all_list, laplacian_list = [], []
    for view in views:
        x_train, x_test, _ = standardize_split(view[tr_idx], view[te_idx])
        x_all = np.concatenate([x_train, x_test], axis=0)
        laplacian, _ = build_graph(x_all, payload["k"])
        x_all_list.append(to_tensor(x_all, device))
        laplacian_list.append(to_tensor(laplacian, device))
    model = MCgnnFull(
        payload["in_dims"], payload["num_classes"], payload["hidden_dims"],
        payload["dropout"], payload["vcdn_hidden"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    with torch.no_grad():
        _, logits, _, embedding = model(x_all_list, laplacian_list, return_embedding=True)
        probability = torch.softmax(logits, dim=1).cpu().numpy()
    n_train = len(tr_idx)
    prediction = probability.argmax(axis=1)
    result = compute_all_metrics(
        y[te_idx], prediction[n_train:], probability[n_train:],
        embedding.cpu().numpy()[n_train:], payload["num_classes"],
    )
    return {"checkpoint": str(checkpoint_path), "dataset": dataset, "test_samples": int(len(te_idx)), "metrics": result}

def cross_validate(view_list, y, n_splits=5, k=10, seed=42, **kwargs):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    test_results = {"Acc": [], "F1_weighted": [], "BACC": []}
    train_results = {"Acc": [], "F1_weighted": [], "BACC": []}
 
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        m_test, m_train, _, _ = train_one_fold(view_list, y, train_idx, test_idx, k=k, **kwargs)
        test_results["Acc"].append(m_test["acc"])
        test_results["F1_weighted"].append(m_test["f1_weighted"])
        test_results["BACC"].append(m_test["recall"])
        train_results["Acc"].append(m_train["acc"])
        train_results["F1_weighted"].append(m_train["f1_weighted"])
        train_results["BACC"].append(m_train["recall"])
        print(f"Fold {fold+1}/{n_splits}  test: Acc={m_test['acc']:.3f}, "
              f"F1_weighted={m_test['f1_weighted']:.3f}")
 
    summary = {k_: (float(np.mean(v)), float(np.std(v))) for k_, v in test_results.items()}
    summary_train = {k_: (float(np.mean(v)), float(np.std(v))) for k_, v in train_results.items()}
    return summary, summary_train

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
 
 
def parse_args():
    p = argparse.ArgumentParser(description="MCgnn: Multiview-Cooperated Graph Neural Network")
    p.add_argument("--data-path", "--data_path", dest="data_path", type=str, default=None)
    p.add_argument("--data-dir", default="dataset", help="Canonical dataset root, or project root containing dataset/.")
    p.add_argument(
        "--dataset", "--type", dest="dataset", type=str, default="BRCA",
        choices=["BRCA", "BRCA-v5", "GBM", "LGG"],
    )
    p.add_argument("--k", type=float, default=10, help="density-threshold scaling hyperparameter")
    p.add_argument("--folds", type=int, default=10, help="StratifiedKFold splits per seed")
    p.add_argument("--seeds", type=int, nargs="+", default=[223, 777, 2026])
    p.add_argument("--output-dir", "--result_path", dest="output_dir", type=str, default="outputs/mcgnn")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--name_file", type=str, default=None,
                    help="log file base name")
    p.add_argument("--overall_log_name", type=str, default=None,
                    help="optional separate file name for the final OVERALL RESULT block")
    p.add_argument("--pretrain-epochs", "--pretrain_epochs", dest="pretrain_epochs", type=int, default=100)
    p.add_argument("--epochs", "--main_epochs", dest="main_epochs", type=int, default=200)
    p.add_argument("--lr-pretrain", "--lr_pretrain", dest="lr_pretrain", type=float, default=1e-3)
    p.add_argument("--lr", "--lr_encoder", dest="lr_encoder", type=float, default=5e-4)
    p.add_argument("--lr-vcdn", "--lr_vcdn", dest="lr_vcdn", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--hidden-dims", "--hidden_dims", dest="hidden_dims", type=int, nargs="+", default=[128, 256, 256])
    p.add_argument("--vcdn-hidden", "--vcdn_hidden", dest="vcdn_hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--checkpoint", default=None, help="Reload this fold checkpoint and evaluate its saved test split.")
    return p.parse_args()
 
 
def main():
    args = parse_args()
    if args.seed is not None:
        args.seeds = [args.seed]
    if args.folds < 2 or args.main_epochs < 1 or args.pretrain_epochs < 0:
        raise ValueError("--folds must be >=2, --main_epochs >=1, and --pretrain_epochs >=0.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    torch.manual_seed(args.seeds[0])
    np.random.seed(args.seeds[0])
 
    args.data_path = args.data_path or str(dataset_mat_path(args.dataset, args.data_dir))

    args.type = args.dataset
    if not Path(args.data_path).is_file():
        raise FileNotFoundError(f"Dataset MAT file not found: {args.data_path}.")
    device = torch.device("cpu") if args.device == "cpu" else get_device()
    if args.checkpoint is not None:
        print(json.dumps(evaluate_checkpoint(args.checkpoint, args.data_path, args.dataset, device), indent=2, default=float))
        return
    gene, methylation, mirna, labels, indexes = load_data(
        args.data_path, mat_dataset_name(args.dataset)
    )
    y, label_map = relabel(labels)
    logging.info("Label mapping: %s", label_map)
    logging.info("Number of classes: %d", len(label_map))
    view_list = build_view_list(gene, methylation, mirna)
    for name, v in zip(VIEW_NAMES, view_list):
        logging.info("View %-16s shape=%s", name, v.shape)
 
    run_name = args.run_name or datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = Path(args.output_dir) / args.dataset.lower() / run_name
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists; prior outputs are not overwritten.")
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    logging.info("Using device: %s", device)
 
    name_file = args.name_file or args.dataset
    logging.info("Run log: %s/%s.log; per-seed logs: %s/seed_logs/seed_<seed>.log", run_dir, name_file, run_dir)
    logging.info("Protocol: %d seeds x StratifiedKFold(%d) = %d total runs",
                 len(args.seeds), args.folds, len(args.seeds) * args.folds)
 
    all_scores, per_seed_scores = run_experiment(
        view_list, y,
        result_path=str(run_dir),
        name_file=name_file,
        seeds=tuple(args.seeds),
        n_splits=args.folds,
        k=args.k,
        overall_log_name=args.overall_log_name,
        num_classes=len(label_map),
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        vcdn_hidden=args.vcdn_hidden,
        pretrain_epochs=args.pretrain_epochs,
        main_epochs=args.main_epochs,
        lr_pretrain=args.lr_pretrain,
        lr_encoder=args.lr_encoder,
        lr_vcdn=args.lr_vcdn,
        device=device,
        checkpoint_dir=str(run_dir / "checkpoints"),
    )
 
    print(f"\n===== MCgnn on {args.type} — "
          f"{len(args.seeds)} seeds x {args.folds}-fold CV =====")
    seed_mean_scores = _seed_mean_scores(per_seed_scores)
    for key in ["acc", "f1", "f1_weighted", "precision", "recall",
                "auc", "prauc", "ari", "mcc", "dbi", "ss"]:
        arr = np.array(seed_mean_scores[key], dtype=float)
        print(f"  {key:>12}: {np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f}")
 
 
if __name__ == "__main__":
    main()


