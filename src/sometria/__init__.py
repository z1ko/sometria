
from typing import Any

import torch as t
import torch.nn as nn
import torch.nn.functional as f

class Expert(nn.Module):
    def __init__(self, hidden_dim: int, expert_dim: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, hidden_dim)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.net(x)
    
class SparseMoE(nn.Module):
    def __init__(self, hidden_dim: int, expert_dim: int, num_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList([
            Expert(hidden_dim, expert_dim)
            for _ in range(num_experts)
        ])

    def forward(self, x: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        batch, sequence, hidden = x.shape
        
        tokens = x.reshape(-1, hidden)
        
        router_logits = self.router(tokens)
        router_probs = f.softmax(router_logits, dim=-1)
        top_probs, top_indices = t.topk(router_probs, self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        output = t.zeros_like(tokens)
        for expert_id, expert in enumerate(self.experts):
            
            token_indices, slot_indices = t.where(top_indices == expert_id)
            if token_indices.numel() == 0:
                continue

            expert_tokens = tokens[token_indices]
            expert_output = expert(expert_tokens)

            weights = top_probs[token_indices, slot_indices].unsqueeze(-1)
            output.index_add_(0, token_indices, expert_output * weights)

        mean_router_prob = router_probs.mean(dim=0)
        selected_fract = f.one_hot(top_indices, num_classes=self.num_experts).float().mean(dim=(0,1))
        load_balance_loss = self.num_experts * t.sum(mean_router_prob * selected_fract)

        return output.reshape(batch, sequence, hidden), load_balance_loss

