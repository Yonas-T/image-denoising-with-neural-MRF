"""Neural message passing layers for the NMRF framework.

Key components:

* Neighbour agregation: multi-head attention over a
  spatial ``M×M`` window with relative positional bias (Eq. 7-8).
* Self agregation: channel-grouped self-attention
  modelling intra-pixel feature competition.
* Message passing layer: one message-passing round
  combining aggregation + feed-forward MLP (Eq. 5-6).
* NeuralMRF: stacked message-passing iterations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .potentials import PottsAggregation


# ------------------------------------------------------------------ #
#  Neighbor aggregation (Eq. 7-8)
# ------------------------------------------------------------------ #

class NeighborAggregation(nn.Module):

    def __init__(
        self, dim: int, num_heads: int = 4, window_size: int = 5
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        # Linear projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Relative positional bias: (num_heads, M*M)
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(num_heads, window_size * window_size)
        )
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute neighbor attention messages.

        B, C, H, W = x.shape
        M = self.window_size
        pad = M // 2
        nh = self.num_heads
        hd = self.head_dim

        # Reflect-pad -> (B, C, H+2p, W+2p)
        x_pad = F.pad(x, [pad, pad, pad, pad], mode="reflect")

        # Extract M×M neighborhoods via unfold -> (B, C, H, W, M, M)
        patches = x_pad.unfold(2, M, 1).unfold(3, M, 1)
        # -> (B, H, W, M*M, C)
        patches = (
            patches.contiguous()
            .reshape(B, C, H, W, M * M)
            .permute(0, 2, 3, 4, 1)
        )

        # Center features -> (B, H, W, C)
        center = x.permute(0, 2, 3, 1)

        # Project Q from center, K/V from patches
        q = self.q_proj(center)   # (B, H, W, C)
        k = self.k_proj(patches)  # (B, H, W, M*M, C)
        v = self.v_proj(patches)  # (B, H, W, M*M, C)

        # Reshape for multi-head attention
        # q: (B, H, W, nh, hd) -> (B*H*W, nh, 1, hd)
        q = q.reshape(B * H * W, nh, hd).unsqueeze(2)
        # k: (B, H, W, M*M, nh, hd) -> (B*H*W, nh, M*M, hd)
        k = k.reshape(B * H * W, M * M, nh, hd).permute(0, 2, 1, 3)
        # v: same layout as k
        v = v.reshape(B * H * W, M * M, nh, hd).permute(0, 2, 1, 3)

        # Attention scores: (B*H*W, nh, 1, M*M)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative positional bias: (1, nh, 1, M*M)
        attn = attn + self.rel_pos_bias.unsqueeze(0).unsqueeze(2)

        attn = F.softmax(attn, dim=-1)

        # Aggregate values: (B*H*W, nh, 1, hd)
        out = attn @ v
        # -> (B*H*W, C)
        out = out.squeeze(2).reshape(B * H * W, C)
        out = self.out_proj(out)

        # -> (B, C, H, W)
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


class SelfAggregation(nn.Module):

    def __init__(
        self, dim: int, num_heads: int = 4, num_groups: int = 8
    ) -> None:
        super().__init__()
        assert dim % num_groups == 0, "dim must be divisible by num_groups"

        self.dim = dim
        self.num_groups = num_groups
        self.group_dim = dim // num_groups

        gd = self.group_dim
        self.qkv = nn.Linear(gd, 3 * gd)
        self.out_proj = nn.Linear(gd, gd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, C, H, W = x.shape
        G = self.num_groups
        gd = self.group_dim

        # (B, C, H, W) -> (B*H*W, G, gd)
        x_r = x.permute(0, 2, 3, 1).reshape(B * H * W, G, gd)

        # QKV for self-attention across groups
        qkv = self.qkv(x_r)  # (B*H*W, G, 3*gd)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B*H*W, G, gd)

        # Attention: (B*H*W, G, G)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(gd)
        attn = F.softmax(attn, dim=-1)

        # Aggregate: (B*H*W, G, gd)
        out = attn @ v
        out = self.out_proj(out)

        # -> (B, C, H, W)
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


class MessagePassingLayer(nn.Module):

    def __init__(self, dim: int, aggregation_module: nn.Module) -> None:
        super().__init__()
        self.aggregation = aggregation_module
        self.norm1 = nn.InstanceNorm2d(dim)
        self.norm2 = nn.InstanceNorm2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )

    def forward(self, mu: torch.Tensor) -> torch.Tensor:

        # Eq. 5: aggregation with residual
        mu_hat = mu + self.aggregation(self.norm1(mu))
        # Eq. 6: MLP with residual
        mu = mu_hat + self.mlp(self.norm2(mu_hat))
        return mu


class NeuralMRF(nn.Module):

    def __init__(
        self,
        dim: int,
        num_iterations: int = 4,
        window_size: int = 5,
        num_heads: int = 4,
        num_groups: int = 8,
        use_self_edges: bool = True,
        use_neural_potentials: bool = True,
    ) -> None:
        super().__init__()
        self.num_iterations = num_iterations
        self.use_self_edges = use_self_edges

        layers: list[nn.Module] = []
        for _ in range(num_iterations):
            # Neighbor aggregation
            if use_neural_potentials:
                neighbor_agg = NeighborAggregation(dim, num_heads, window_size)
            else:
                neighbor_agg = PottsAggregation(dim, window_size)
            layers.append(MessagePassingLayer(dim, neighbor_agg))

            # Self-edge aggregation (optional)
            if use_self_edges:
                self_agg = SelfAggregation(dim, num_heads, num_groups)
                layers.append(MessagePassingLayer(dim, self_agg))

        self.layers = nn.ModuleList(layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mu = features
        for layer in self.layers:
            mu = layer(mu)
        return mu
