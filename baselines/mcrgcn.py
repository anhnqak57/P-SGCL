from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


class Contrast(nn.Module):
    """Directional cross-view contrastive objective retained from MCRGCN."""

    def __init__(self, hidden_dim: int, tau: float = 0.5, lam: float = 0.5) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.tau = tau
        self.lam = lam
        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.414)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        return torch.exp(torch.mm(z1, z2.t()) / (z1_norm * z2_norm.t() + 1e-8) / self.tau)

    def forward(
        self,
        z_ge: torch.Tensor,
        z_mp: torch.Tensor,
        z_sc: torch.Tensor,
        positives: torch.Tensor,
    ) -> torch.Tensor:
        z_proj_ge = self.proj(z_ge)
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)

        gene_methy = self.sim(z_proj_ge, z_proj_mp)
        methy_gene = gene_methy.t()
        gene_methy = gene_methy / (gene_methy.sum(dim=1, keepdim=True) + 1e-8)
        methy_gene = methy_gene / (methy_gene.sum(dim=1, keepdim=True) + 1e-8)
        loss_gene_methy = (
            self.lam * -torch.log(gene_methy.mul(positives).sum(dim=-1)).mean()
            + (1 - self.lam) * -torch.log(methy_gene.mul(positives).sum(dim=-1)).mean()
        )

        gene_mirna = self.sim(z_proj_ge, z_proj_sc)
        mirna_gene = gene_mirna.t()
        gene_mirna = gene_mirna / (gene_mirna.sum(dim=1, keepdim=True) + 1e-8)
        mirna_gene = mirna_gene / (mirna_gene.sum(dim=1, keepdim=True) + 1e-8)
        loss_gene_mirna = (
            self.lam * -torch.log(gene_mirna.mul(positives).sum(dim=-1)).mean()
            + (1 - self.lam) * -torch.log(mirna_gene.mul(positives).sum(dim=-1)).mean()
        )
        return loss_gene_methy + loss_gene_mirna


class GCNModel(nn.Module):
    """One MCRGCN view encoder."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.gcn1 = ChebConv(num_features, hidden_dim, K=2)
        self.gcn2 = ChebConv(hidden_dim, hidden_dim, K=2)
        self.gcn3 = ChebConv(hidden_dim, output_dim, K=2)
        self.dropout = nn.Dropout(dropout)
        self.linear_projection = nn.Linear(num_features, hidden_dim)
        self.norm1 = nn.LayerNorm(num_features, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, data: object) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        residual = self.linear_projection(x)
        x = F.silu(self.gcn1(self.dropout(self.norm1(x)), edge_index))
        x = F.silu(residual + self.gcn2(self.dropout(self.norm2(x)), edge_index))
        return F.silu(self.gcn3(x, edge_index))


class HeCo(nn.Module):
    """Three-view MCRGCN encoder."""

    def __init__(
        self,
        num_feature1: int,
        num_feature2: int,
        num_feature3: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.ge = GCNModel(num_feature1, hidden_dim, output_dim, dropout)
        self.mp = GCNModel(num_feature2, hidden_dim, output_dim, dropout)
        self.sc = GCNModel(num_feature3, hidden_dim, output_dim, dropout)
        self.projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode(self, encoder: nn.Module, data: object) -> torch.Tensor:
        return self.projector(encoder(data))

    def forward(self, data1: object, data2: object, data3: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.encode(self.ge, data1),
            self.encode(self.mp, data2),
            self.encode(self.sc, data3),
        )

    @torch.no_grad()
    def get_embeds(self, data1: object, data2: object, data3: object):
        z_ge, _, _ = self.forward(data1, data2, data3)
        return z_ge.detach().cpu().numpy()
