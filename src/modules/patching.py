import torch
import torch.nn as nn

class patchEmbed(nn.Module):
    def __init__(self, token_dim, patch_size, img_size=32, in_channels=3) -> None:
        super().__init__()
        # This one layer does both [patching] and [linear projection]
        self.proj = nn.Conv2d(in_channels, token_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)# (B, 192,8,8)
        x = x.flatten(2)
        x = x.transpose(1,2) #(B,64,192)
        return x