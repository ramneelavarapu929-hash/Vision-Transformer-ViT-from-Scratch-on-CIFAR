import torch
import torch.nn as nn

class mlp_head(nn.Module):
    def __init__(self, token_dim, num_classes) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(token_dim)
        self.mlp = nn.Linear(token_dim, num_classes)

    def forward(self, x):
        
        x = self.layernorm(x)
        x = self.mlp(x)
        
        return x