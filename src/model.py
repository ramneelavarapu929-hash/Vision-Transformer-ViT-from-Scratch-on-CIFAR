from .modules.patching import patchEmbed
from .modules.transformer import transformer_encoder
from .modules.mlp import mlp_head

import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, token_dim, num_classes, patch_size, num_patches, transformer_blocks) -> None:
        super().__init__()
        self.patch_embedding = patchEmbed(token_dim, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1,1,token_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches+1, token_dim))
        self.transformer_blocks = nn.Sequential(*[transformer_encoder(token_dim) for _ in range(transformer_blocks)])
        self.mlp_head = mlp_head(token_dim, num_classes)
    
    def forward(self, x):
        
        x = self.patch_embedding(x)
        B = x.shape[0]

        cls_token = self.cls_token.expand(B, -1,-1)
        x = torch.cat((cls_token,x), dim =1)
        x = x + self.position_embedding

        x = self.transformer_blocks(x)
        x = x[:,0]
        x = self.mlp_head(x)

        return x
