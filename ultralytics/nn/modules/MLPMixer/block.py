import torch
from torch import nn
from torch.nn import functional as F
from ..conv import *

__all__ = ["MLPMixer"]

class MLPMixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MLPMixerLayer, self).__init__()
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)

        self.token_mixing = nn.Sequential(
            nn.Linear(num_patches, tokens_mlp_dim),
            nn.GELU(),
            nn.Linear(tokens_mlp_dim, num_patches),
        )

        self.channel_mixing = nn.Sequential(
            nn.Linear(hidden_dim, channels_mlp_dim),
            nn.GELU(),
            nn.Linear(channels_mlp_dim, hidden_dim),
        )

    def forward(self, x):
        # Layer normalization
        y = self.layernorm1(x)
        # Token mixing
        y = self.token_mixing(y.transpose(1, 2)).transpose(1, 2)
        # y = self.token_mixing(y)
        # Residual connection
        x = x + y
        # Layer normalization
        y = self.layernorm2(x)
        # Channel mixing
        y = self.channel_mixing(y)
        # Residual connection
        x = x + y
        return x


class MLPMixer(nn.Module):
    def __init__(self, c1, c2, image_size, patch_size, hidden_dim=256,  num_layers=2, tokens_mlp_dim=256, channels_mlp_dim=256):
        super(MLPMixer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        self.patch_embed = nn.Conv2d(c1, hidden_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.mixer_layers = nn.ModuleList([
            MLPMixerLayer(num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim)
            for _ in range(num_layers)
        ])
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.conv = Conv(hidden_dim, c2, 3, 1)
        # self.head = nn.Linear(hidden_dim, c2)

    def forward(self, x):
        
        # print("输入图像的尺寸为:", x.shape)  
        b, _, w, h = x.shape     
        x = self.patch_embed(x)  # [B, C, H, W]
        # b2, c2, _, _ = x.shape
        # x = torch.Tensor.resize(x, [b2, c2, w // self.patch_size, h // self.patch_size])
        
        
        x = x.flatten(2)  # Flatten height and width into patches
        x = x.transpose(1, 2)  # [B, num_patches, hidden_dim]

        

        for layer in self.mixer_layers:
            x = layer(x)

        x = self.layernorm(x)
        x = x.transpose(1, 2).reshape(b, -1,  w // self.patch_size, h // self.patch_size)
        x = self.conv(x)
        # print("输图像的尺寸为:", x.shape)
        # x = x.mean(dim=1)  # Global average pooling over patches
        # x = self.head(x)
        
        return x
