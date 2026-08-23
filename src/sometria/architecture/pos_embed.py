import torch as t
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, num_time_patches: int, num_dofs_patches: int, embed_dim: int) -> None:
        super().__init__()

        self.num_time_patches = num_time_patches
        self.num_dofs_patches = num_dofs_patches
        self.embed_dim = embed_dim

        self.time_encoding = nn.Parameter(t.zeros((1, self.num_time_patches, 1, self.embed_dim)))
        nn.init.trunc_normal_(self.time_encoding, std=0.02)

        self.dofs_encoding = nn.Parameter(t.zeros((1, 1, self.num_dofs_patches, self.embed_dim)))
        nn.init.trunc_normal_(self.dofs_encoding, std=0.02)

    def get_grid(self) -> t.Tensor:
        return self.time_encoding + self.dofs_encoding # [1, T, D, E]

    def get_flat(self) -> t.Tensor:
        return self.get_grid().flatten(1, 2) # [1, T*D, E]

    def add_flat(self, x: t.Tensor, idx: t.Tensor | None = None) -> t.Tensor:
        if idx is not None:
            return x + self.gather(idx)
        return x + self.get_flat()

    def gather(self, idx: t.Tensor) -> t.Tensor:
        idx = idx.to(device=self.time_encoding.device, dtype=t.long)
        pos = self.get_flat().expand(idx.shape[0], -1, -1)
        idx = idx.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        return pos.gather(dim=1, index=idx)