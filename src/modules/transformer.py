import torch
import torch.nn as nn

mlp_hidden_dim = 64
num_heads = 8

class transformer_encoder(nn.Module):
    def __init__(self, token_dim) -> None:
        super().__init__()
        self.layernorm1 = nn.LayerNorm(token_dim)
        self.layernorm2 =nn.LayerNorm(token_dim)
        self.multiheadattention = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, token_dim)

        )

    def forward(self, x):
        residual1 = x

        x = self.layernorm1(x)
        x = self.multiheadattention(x,x,x)[0]
        x= x + residual1

        residual2 =x

        x = self.layernorm2(x)
        x= self.mlp(x)
        x = x + residual2

        return x