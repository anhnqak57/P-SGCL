from __future__ import annotations

import logging
import numpy as np
import scipy.io as sio
import random
import torch
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PERCENTILE_LIST = np.arange(91, 100, 0.5)
def corr_x_y(x: np.ndarray, y: np.ndarray, eps: float=1e-8) -> np.ndarray:
    assert x.shape[1] == y.shape[1], "Different shape"
    x = x - np.mean(x, axis=1, keepdims=True)
    y = y - np.mean(y, axis=1, keepdims=True)
    lxy = np.dot(x, y.T)
    lxx = np.diag(np.dot(x, x.T)).reshape((-1, 1))
    lyy = np.diag(np.dot(y, y.T)).reshape((1, -1))
    corr = lxy / (np.dot(np.sqrt(lxx), np.sqrt(lyy)) + 1e-8)
    return corr


def load_data(data_path: str="dataset/brca/source/BRCA.mat", type='BRCA'):
    data = sio.loadmat(data_path)
    gene = data[f'{type}_Gene_Expression'].T
    methylation = data[f'{type}_Methy_Expression'].T
    mirna = data[f'{type}_Mirna_Expression'].T
    labels = data[f'{type}_clinicalMatrix'].reshape(-1)
    indexes = data[f'{type}_indexes'].flatten()
    logging.info("Shape of Gene Expression: %s", gene.shape)
    logging.info("Shape of Methylation Expression: %s", methylation.shape)
    logging.info("Shape of miRNA Expression: %s", mirna.shape)
    logging.info("Shape of labels: %s", labels.shape)
    logging.info("Shape of indexes: %s", indexes.shape)
    return gene, methylation, mirna, labels, indexes

def set_seed(seed:int =777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def load_edges(file_path, node_dict):
    edge_indexes = []
    with open(file_path, 'r') as f:
        for line in f:
            node1, node2 = line.strip().split(',')
            if node1 in node_dict and node2 in node_dict:
                node1_id = node_dict[node1]
                node2_id = node_dict[node2]
                edge_indexes.append([node1_id, node2_id])
                edge_indexes.append([node2_id, node1_id])
    edge_indexes = torch.tensor(edge_indexes, dtype=torch.long).t().contiguous()
    return edge_indexes


def convert_csv_to_mat(data_dir, type="BRCA"):
    labels_path = os.path.join(data_dir, "labels.csv")
    gene_path = os.path.join(data_dir, f"{type}_mRNA_top.csv")
    miRNA_path = os.path.join(data_dir, f"{type}_miRNA_top.csv")
    methy_path = os.path.join(data_dir, f"{type}_Methy_top.csv")
    
    gene_data = pd.read_csv(gene_path, index_col=0).T
    miRNA_data = pd.read_csv(miRNA_path, index_col=0).T
    methy_data = pd.read_csv(methy_path, index_col=0).T
    labels = pd.read_csv(labels_path).values.flatten()
    indexes = gene_data.index.values
    
    assert set(methy_data.index) == set(gene_data.index) == set(miRNA_data.index) == set(indexes)
    indexes_gene_dict = {idx: i for i, idx in enumerate(indexes)}
    gene_data.index = gene_data.index.map(indexes_gene_dict)
    methy_data.index = methy_data.index.map(indexes_gene_dict)
    miRNA_data.index = miRNA_data.index.map(indexes_gene_dict)
    
    save_mat_path = os.path.join(data_dir, f"{type}_labels.mat")
    sio.savemat(save_mat_path, {
        f'{type}_Gene_Expression': gene_data.values.T,
        f'{type}_Methy_Expression': methy_data.values.T,
        f'{type}_Mirna_Expression': miRNA_data.values.T,
        f'{type}_clinicalMatrix': labels.reshape(-1, 1),
        f'{type}_indexes': miRNA_data.index.values.reshape(-1, 1),
    })

def load_data_csv(data_folder ="./data2/BRCA_v2", save_mat_path=None):
    labels_path = os.path.join(data_folder, "labels.csv")
    labels_path = os.path.join(data_folder, "labels.csv")
    BRCA_gene_path = os.path.join(data_folder, "BRCA_mRNA_top.csv")
    BRCA_miRNA_path = os.path.join(data_folder, "BRCA_miRNA_top.csv")
    BRCA_Methy_path = os.path.join(data_folder, "BRCA_Methy_top.csv")
    gene_data = pd.read_csv(BRCA_gene_path, index_col=0).T
    methy_data = pd.read_csv(BRCA_Methy_path, index_col=0).T
    miRNA_data = pd.read_csv(BRCA_miRNA_path, index_col=0).T
    labels = pd.read_csv(labels_path).values.flatten()
    indexes = gene_data.index.values
    logging.info("Shape of Gene Expression: %s", gene_data.shape)
    logging.info("Shape of Methylation Expression: %s", methy_data.shape)
    logging.info("Shape of miRNA Expression: %s", miRNA_data.shape)
    logging.info("Shape of labels: %s", labels.shape)
    logging.info("Shape of indexes: %s", indexes.shape)
    if save_mat_path is not None:
        sio.savemat(save_mat_path, {
            'BRCA_Gene_Expression': gene_data.values.T,
            'BRCA_Methy_Expression': methy_data.values.T,
            'BRCA_Mirna_Expression': miRNA_data.values.T,
            'BRCA_clinicalMatrix': labels.reshape(-1, 1),
            'BRCA_indexes': indexes.reshape(-1, 1),
        })
        logging.info("Saved .mat file to %s", save_mat_path)
    return gene_data.values, methy_data.values, miRNA_data.values, labels, indexes

def get_percentile_threshold(
    features: np.ndarray,
    method: str = "corr",
    percentile: float = 97,
):
    sim_fn = {
        'corr': lambda x: np.abs(corr_x_y(x, x)),
    }.get(method)
    if sim_fn is None:
        raise ValueError(f"Unknown method: {method!r}. "
                         f"Choose from: corr, cos, euclidean, mahalanobis")
    
    z_sim = sim_fn(features).clip(0, 1)
    off_diag = z_sim[~np.eye(z_sim.shape[0], dtype=bool)]
    threshold = np.percentile(off_diag, percentile)
    return threshold


def build_percentile_graph(
    features: np.ndarray,
    type: str = "BRCA",
    method: str = "corr",
    f_name: str | os.PathLike[str] = './data2/PER_BRCA_',
) -> tuple[np.ndarray, float, float, Path]:
    output_dir = Path(f_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = StandardScaler().fit_transform(features)
    threshold_percentile_list = [get_percentile_threshold(features, method=method, percentile=p) for p in PERCENTILE_LIST]

    sim_fn = {
        'corr':        lambda x: np.abs(corr_x_y(x, x)),
    }.get(method)
    if sim_fn is None:
        raise ValueError(f"Unknown method: {method!r}. "
                         f"Choose from: corr, cos, euclidean, mahalanobis")
    deltas = [abs((threshold_percentile_list[0] + threshold_percentile_list[-1]) / 2 - threshold_percentile_list[i]) for i in range(1, len(threshold_percentile_list))]
    index_choosen = np.argmin(deltas) + 1
    choosen_percentile = PERCENTILE_LIST[index_choosen]
    percentile_threshold = threshold_percentile_list[index_choosen]
    logging.info("Choosen Percentile: %.1f%%", choosen_percentile)
    logging.info("Percentile Threshold: %.4f", percentile_threshold)
    z_sim = sim_fn(features).clip(0, 1)
    normalize_z = np.zeros_like(z_sim, dtype=int)
    for i in range(z_sim.shape[0]):
        row = z_sim[i].copy()
        candidates = row > percentile_threshold
        normalize_z[i, candidates] = 1
    logging.info("Average degree: %.2f", np.sum(normalize_z) / normalize_z.shape[0])
    logging.info ("Min degree: %d", np.min(np.sum(normalize_z, axis=1)))
    logging.info ("Max degree: %d", np.max(np.sum(normalize_z, axis=1)))
    logging.info("Nodes with zero degree: %d", len(np.where(np.sum(normalize_z, axis=1) == 0)[0]))
    matrix_path = output_dir / f'{type}_matrix.mat'
    sio.savemat(matrix_path, {'normalize_corr': normalize_z})
    logging.info("Saved adjacency matrix for %s to %s", type, matrix_path)
    return normalize_z, float(choosen_percentile), float(percentile_threshold), matrix_path
