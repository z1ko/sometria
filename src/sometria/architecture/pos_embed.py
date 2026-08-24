"""Learned position for one token in the `time x DOF` grid.

Time and DOF are separate parameters that broadcast into a grid rather than one
embedding per flat position: a token's identity is "this DOF, this many patches in",
and factoring it that way lets a shorter window reuse a prefix of the time axis
instead of needing its own table.
"""

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

    def get_grid(self, num_time_patches: int | None = None) -> t.Tensor:
        """``[1, TP, D, E]``, sliced to the first ``num_time_patches`` time steps.

        A downstream window may be shorter than the one the encoding was built for. It
        takes the *prefix* of the time axis, so patch 0 keeps meaning "start of window"
        and pretrained embeddings transfer.
        """

        if num_time_patches is None:
            num_time_patches = self.num_time_patches
        if num_time_patches > self.num_time_patches:
            raise ValueError(
                f"asked for {num_time_patches} time patches, "
                f"but the encoding holds {self.num_time_patches}"
            )
        return self.time_encoding[:, :num_time_patches] + self.dofs_encoding

    def get_flat(self, num_time_patches: int | None = None) -> t.Tensor:
        return self.get_grid(num_time_patches).flatten(1, 2) # [1, TP*D, E]

    def add_flat(
        self,
        x: t.Tensor,
        idx: t.Tensor | None = None,
        num_time_patches: int | None = None,
    ) -> t.Tensor:
        if idx is not None:
            return x + self.gather(idx, num_time_patches)
        return x + self.get_flat(num_time_patches)

    def gather(self, idx: t.Tensor, num_time_patches: int | None = None) -> t.Tensor:
        idx = idx.to(device=self.time_encoding.device, dtype=t.long)
        pos = self.get_flat(num_time_patches).expand(idx.shape[0], -1, -1)
        idx = idx.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        return pos.gather(dim=1, index=idx)
