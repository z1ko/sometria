
from typing import Any

import torch as t
import torch.nn as nn

class MotionPatchEmbedding(nn.Module):
    def __init__(
        self,
        num_dofs: int,
        num_features: int = 5,
        patch_size: int = 10,
        embed_dim: int = 256,
        max_patches: int = 128 
    ) -> None:
        super().__init__()

        self.num_dofs = num_dofs
        self.num_features = num_features
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.projection = nn.Linear(patch_size * num_features, embed_dim)
        self.time_embedding = nn.Parameter(t.randn(1, max_patches, 1, embed_dim) * 0.02)
        self.dof_embedding  = nn.Parameter(t.randn(1, 1, num_dofs, embed_dim) * 0.02)

    def forward(self, x: t.Tensor) -> t.Tensor:

        B, T, D, C = x.shape
        num_patches = T // self.patch_size

        if D != self.num_dofs:
            raise ValueError(f"Expected {self.num_dofs} DoFs, got {D}")
        
        if C != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {C}")

        x = x.reshape(B, num_patches, self.patch_size, D, C)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.flatten(start_dim=-2)

        tokens = self.projection(x)
        tokens = (
            tokens
            + self.time_embedding[:, :num_patches]
            + self.dof_embedding
        )

        return tokens



