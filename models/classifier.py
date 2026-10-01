import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VCDN (nn.Module):
    def __init__(
        self,
        num_views: int,
        num_classes: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        in_dim = num_classes ** num_views
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, prob_list):
        x = prob_list[0]
        for p in prob_list[1:]:
            x = torch.einsum("bi,bj->bij", x, p).reshape(x.size(0), -1)
        return self.net(x)
    
class Classifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list = [60, 128],
        vcdn_dim: int = 128,
        num_views: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.vcdn_dim = vcdn_dim
        self.dropout = dropout
        self.num_views = num_views
        
        self.vcdn = VCDN(
            num_views=num_views,
            num_classes=num_classes,
            hidden_dim=vcdn_dim
        )

        self.classifier = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(input_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[-1], num_classes),
            )
            for _ in range(num_views)
            ]
        )
       

    def forward(self, gene_features, methyl_features, mirna_features):
        prob_list = []
        for i, features in enumerate([gene_features, methyl_features, mirna_features]):
            prob = F.softmax(self.classifier[i](features), dim=1)
            prob_list.append(prob)
        sig_list = [torch.sigmoid(prob) for prob in prob_list]
        logits = self.vcdn(sig_list)
        return logits