import torch

from ConvNeXt import *
from ResNet_18 import *
from CrossAttentionFusion import *


class FusedModel(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.convnext = ConvNeXt_Features()
        self.resnet18 = ResNet18_Features()

        self.resnet_proj = nn.Linear(512, embed_dim)

        self.cross_attn_c2r = CrossAttentionFusion(embed_dim)
        self.cross_attn_r2c = CrossAttentionFusion(embed_dim)

        self.weights = nn.Parameter(torch.ones(2))

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x1, x2):
        f1 = self.convnext(x1)
        f2 = self.resnet18(x2)
        f2 = self.resnet_proj(f2)

        f_attn1 = self.cross_attn_c2r(f1, f2)
        f_attn2 = self.cross_attn_r2c(f2, f1)

        norm_weights = torch.softmax(self.weights, dim=0)

        fused = norm_weights[0] * f_attn1 + norm_weights[1] * f_attn2

        fused = fused + f1

        fused = self.norm(fused)
        fused = self.dropout(fused)

        out = self.fc(fused)

        return out
