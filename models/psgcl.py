import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_model import BernGraphModel, GCNModel, GraphModel

class MultiContrastiveLoss(nn.Module):
    def __init__(self, tau=0.5, eps=1e-8):
        super().__init__()
        self.tau = tau
        self.eps = eps
        
    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        sim_matrix = torch.exp(torch.mm(z1, z2.t()) / (z1_norm * z2_norm.t() + self.eps) / self.tau)
        return sim_matrix

    def forward(self, z_gene, z_methylation, z_mirna, z_pos):
        matrix_gene2gene = self.sim(z_gene, z_gene)
        matrix_gene2gene = matrix_gene2gene / (torch.sum(matrix_gene2gene, dim=1).view(-1, 1) + self.eps)
        self_loss_gene = -torch.log(matrix_gene2gene.mul(z_pos).sum(dim=-1)).mean()
        matrix_methylation2methylation = self.sim(z_methylation, z_methylation)
        matrix_methylation2methylation = matrix_methylation2methylation / (torch.sum(matrix_methylation2methylation, dim=1).view(-1, 1) + self.eps)
        self_loss_methylation = -torch.log(matrix_methylation2methylation.mul(z_pos).sum(dim=-1)).mean()
        matrix_mirna2mirna = self.sim(z_mirna, z_mirna)
        matrix_mirna2mirna = matrix_mirna2mirna / (torch.sum(matrix_mirna2mirna, dim=1).view(-1, 1) + self.eps)
        self_loss_mirna = -torch.log(matrix_mirna2mirna.mul(z_pos).sum(dim=-1)).mean()
        self_loss = self_loss_methylation + self_loss_mirna + self_loss_gene




        return self_loss

class PSGCL(nn.Module):
    def __init__(
        self,
        num_features1,
        num_features2,
        num_features3,
        hidden_dim=256,
        output_dim=128,
        dropout=0.3,
        K=2
    ):
        super().__init__()
        self.gene = GraphModel(num_features1, hidden_dim, output_dim, dropout, K=K)
        self.methylation = GraphModel(num_features2, hidden_dim, output_dim, dropout, K=K)
        self.mirna = GraphModel(num_features3, hidden_dim, output_dim, dropout, K=K)







        self.ge_projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.methylation_projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.mirna_projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )


    def encode(self, encoder, projector, data):
        z = encoder(data)
        z = projector(z)
        return z

    def forward(self, data1, data2, data3):
        z_gene = self.encode(self.gene, self.ge_projector, data1)
        z_methylation = self.encode(self.methylation, self.methylation_projector, data2)
        z_mirna = self.encode(self.mirna, self.mirna_projector, data3)


        return z_gene, z_methylation, z_mirna
    
    @torch.no_grad()
    def get_embeds(self, data1, data2, data3, alpha1=1.0, alpha2=1.0, alpha3=1.0):
        z_gene = self.encode(self.gene, self.ge_projector, data1)
        z_methylation = self.encode(self.methylation, self.methylation_projector, data2)
        z_mirna = self.encode(self.mirna, self.mirna_projector, data3)

        z = (alpha1 * z_gene + alpha2 * z_methylation + alpha3 * z_mirna) / (alpha1 + alpha2 + alpha3)



        return z.detach().cpu().numpy()
    

    @torch.no_grad()
    def get_embeds_per_omics(self, data1, data2, data3, alpha1=1.0, alpha2=1.0, alpha3=1.0):
        z_gene = self.encode(self.gene, self.ge_projector, data1)
        z_methylation = self.encode(self.methylation, self.methylation_projector, data2)
        z_mirna = self.encode(self.mirna, self.mirna_projector, data3)

        return z_gene, z_methylation, z_mirna
    
class PSGCLWithoutContrastive(nn.Module):
    def __init__(
        self,
        num_features1,
        num_features2,
        num_features3,
        hidden_dim=256,
        output_dim=128,
        dropout=0.3,
        num_classes=4,
        hidden_dim1=60,
        hidden_dim2=30
    ):
        super().__init__()
        self.gene = GraphModel(num_features1, hidden_dim, output_dim, dropout)
        self.methylation = GraphModel(num_features2, hidden_dim, output_dim, dropout)
        self.mirna = GraphModel(num_features3, hidden_dim, output_dim, dropout)

        self.ge_projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.methylation_projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.mirna_projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, num_classes)
        )
    def encode(self, encoder, projector, data):
        z = encoder(data)
        z = projector(z)
        return z
    

    def forward(self, data1, data2, data3):
        z_gene = self.encode(self.gene, self.ge_projector, data1)
        z_methylation = self.encode(self.methylation, self.methylation_projector, data2)
        z_mirna = self.encode(self.mirna, self.mirna_projector, data3)

        z = (z_gene + z_methylation + z_mirna) / 3.0
        logits = self.classifier(z)

        return logits
    
    @torch.no_grad()
    def get_embeds(self, data1, data2, data3, alpha1=1.0, alpha2=1.0, alpha3=1.0):
        z_gene = self.encode(self.gene, self.ge_projector, data1)
        z_methylation = self.encode(self.methylation, self.methylation_projector, data2)
        z_mirna = self.encode(self.mirna, self.mirna_projector, data3)

        z = (alpha1 * z_gene + alpha2 * z_methylation + alpha3 * z_mirna) / (alpha1 + alpha2 + alpha3)
        return z.detach().cpu().numpy()
    
    @torch.no_grad()
    def get_embeds_per_omics(self, data1, data2, data3, alpha1=1.0, alpha2=1.0, alpha3=1.0):
        z_gene = self.encode(self.gene, self.ge_projector, data1)
        z_methylation = self.encode(self.methylation, self.methylation_projector, data2)
        z_mirna = self.encode(self.mirna, self.mirna_projector, data3)

        z = (alpha1 * z_gene + alpha2 * z_methylation + alpha3 * z_mirna) / (alpha1 + alpha2 + alpha3)
        return z.detach().cpu().numpy()