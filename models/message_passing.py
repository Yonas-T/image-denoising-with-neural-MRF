"""Neural message passing layers for the NMRF framework.

Implements the core message-passing operations described in
Guan et al. (CVPR 2024), Equations 5–8, adapted from stereo
matching to 2D image denoising.

Key components:

* :class:`NeighborAggregation` — multi-head attention over a
  spatial ``M×M`` window with relative positional bias (Eq. 7-8).
* :class:`SelfAggregation` — channel-grouped self-attention
  modelling intra-pixel feature competition.
* :class:`MessagePassingLayer` — one message-passing round
  combining aggregation + feed-forward MLP (Eq. 5-6).
* :class:`NeuralMRF` — stacked message-passing iterations.
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
    """Attentional aggregation over an M×M spatial neighborhood.

    For every pixel, queries are formed from the center feature and
    keys/values from the surrounding ``window_size × window_size``
    patch.  Multi-head attention is used with a learnable relative
    positional bias table (one scalar per relative offset per head).

    Args:
        dim: Feature channel dimension ``C``.
        num_heads: Number of attention heads.
        window_size: Side length ``M`` of the square window (odd).
    """

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
        """Compute neighbor attention messages.

        Args:
            x: Feature map of shape ``(B, C, H, W)``.

        Returns:
            Message tensor of shape ``(B, C, H, W)``.
        """
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


# ------------------------------------------------------------------ #
#  Self-edge aggregation
# ------------------------------------------------------------------ #

class SelfAggregation(nn.Module):
    """Channel-grouped self-attention for intra-pixel competition.

    The feature channels are split into ``num_groups`` groups.  Within
    each spatial location, self-attention is computed *across* the
    groups (treating each group vector as a token).  This models
    competition among different feature "hypotheses" at every pixel.

    Args:
        dim: Feature dimension ``C`` (must be divisible by *num_groups*).
        num_heads: Number of attention heads (unused in the current
            per-group QKV scheme but kept for API consistency).
        num_groups: Number of groups ``G``.
    """

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
        """Compute self-edge messages via grouped self-attention.

        Args:
            x: Feature map of shape ``(B, C, H, W)``.

        Returns:
            Message tensor of shape ``(B, C, H, W)``.
        """
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


# ------------------------------------------------------------------ #
#  Message-passing layer (Eq. 5-6)
# ------------------------------------------------------------------ #

class MessagePassingLayer(nn.Module):
    """Single message-passing round (Eq. 5-6 of the paper).

    Comprises an aggregation step (neighbor or self) with a residual
    connection, followed by a 2-layer MLP (expand → GELU → project)
    with another residual.  Instance normalization is applied before
    each sub-layer.

    Args:
        dim: Feature dimension ``C``.
        aggregation_module: An instantiated aggregation module
            (:class:`NeighborAggregation`, :class:`SelfAggregation`,
            or :class:`PottsAggregation`).
    """

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
        """Run one message-passing iteration.

        Args:
            mu: Current belief tensor of shape ``(B, C, H, W)``.

        Returns:
            Updated belief tensor of shape ``(B, C, H, W)``.
        """
        # Eq. 5: aggregation with residual
        mu_hat = mu + self.aggregation(self.norm1(mu))
        # Eq. 6: MLP with residual
        mu = mu_hat + self.mlp(self.norm2(mu_hat))
        return mu


# ------------------------------------------------------------------ #
#  Full Neural MRF
# ------------------------------------------------------------------ #

class NeuralMRF(nn.Module):
    """Stacked neural message-passing module.

    Iteratively refines a coarse feature map through alternating
    neighbor-edge and (optionally) self-edge aggregation layers.

    Args:
        dim: Feature dimension (typically ``4 × base_channels = 128``).
        num_iterations: Number of message-passing rounds.
        window_size: Spatial window size ``M`` for neighbor aggregation.
        num_heads: Attention heads in neighbor aggregation.
        num_groups: Groups in self-edge aggregation.
        use_self_edges: Whether to include self-edge layers.
        use_neural_potentials: If *True*, use learned
            :class:`NeighborAggregation`; if *False*, fall back to
            :class:`PottsAggregation`.
    """

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
        """Iteratively refine features via message passing.

        Args:
            features: Coarse feature map of shape ``(B, C, H, W)``.

        Returns:
            Refined feature map of the same shape.
        """
        mu = features
        for layer in self.layers:
            mu = layer(mu)
        return mu
