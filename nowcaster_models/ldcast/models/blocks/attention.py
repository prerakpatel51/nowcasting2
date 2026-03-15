# Copied from LDCast paper: ldcast/models/blocks/attention.py. No modifications.

"""Temporal attention and positional encoding modules for the ldcast forecaster.

These modules implement the temporal transformer (Stage 2 of the forecaster),
which maps from Din input time steps to Dout output time steps using
cross-attention along the time dimension only (spatial dims are independent).

The key insight is that attention is computed per spatial location (h,w) across
the time dimension, so the attention matrix is Dq x Dk (typically small, e.g. 5x1)
rather than (H*W) x (H*W) which would be prohibitively expensive.

Classes:
    TemporalAttention    - Multi-head attention along the temporal dimension only.
    TemporalTransformer  - Full transformer block: self-attn + cross-attn + MLP.
    MLP                  - Simple 2-layer feed-forward network.

Functions:
    positional_encoding  - Sinusoidal positional encoding for time coordinates.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Multi-head attention computed only along the temporal (D) dimension.

    For each spatial location (h, w), computes attention across time steps.
    This is efficient because the attention matrix is Dq x Dk per location,
    where D is typically very small (e.g., 1-5 time steps).

    Supports both self-attention (y=None, attends to itself) and
    cross-attention (y provided, queries from x attend to keys/values from y).

    Args:
        channels:         Channel dimension of the query input x.
        context_channels: Channel dimension of the key/value input y (default: same as channels).
        head_dim:         Dimension per attention head (default: 32).
        num_heads:        Number of attention heads (default: 8).
    """
    def __init__(
        self, channels, context_channels=None,
        head_dim=32, num_heads=8
    ):
        super().__init__()
        self.channels = channels
        if context_channels is None:
            context_channels = channels
        self.context_channels = context_channels
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.inner_dim = head_dim * num_heads        # Total attention dimension
        self.attn_scale = self.head_dim ** -0.5      # 1/sqrt(d_k) scaling factor
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        # Key and Value projections (from context y)
        self.KV = nn.Linear(context_channels, self.inner_dim*2)
        # Query projection (from input x)
        self.Q = nn.Linear(channels, self.inner_dim)
        # Output projection back to channel dimension
        self.proj = nn.Linear(self.inner_dim, channels)

    def forward(self, x, y=None):
        """Compute temporal attention.

        Args:
            x: Query tensor of shape (B, Dq, H, W, C).
               Dq = number of output time steps.
            y: Key/Value tensor of shape (B, Dk, H, W, C_ctx).
               Dk = number of input time steps. If None, uses x (self-attention).

        Returns:
            Output tensor of shape (B, Dq, H, W, C).
        """
        if y is None:
            y = x  # Self-attention: query = key = value source

        # Project y to keys and values, split into two halves
        (K,V) = self.KV(y).chunk(2, dim=-1)  # Each: (B, Dk, H, W, inner_dim)
        (B, Dk, H, W, C) = K.shape
        # Reshape to multi-head: (B, Dk, H, W, num_heads, head_dim)
        shape = (B, Dk, H, W, self.num_heads, self.head_dim)
        K = K.reshape(shape)
        V = V.reshape(shape)

        # Project x to queries
        Q = self.Q(x)  # (B, Dq, H, W, inner_dim)
        (B, Dq, H, W, C) = Q.shape
        shape = (B, Dq, H, W, self.num_heads, self.head_dim)
        Q = Q.reshape(shape)  # (B, Dq, H, W, num_heads, head_dim)

        # Rearrange for batched matmul: attention is computed per (b, h, w, head)
        K = K.permute((0,2,3,4,5,1))  # (B, H, W, num_heads, head_dim, Dk) -- K transposed
        V = V.permute((0,2,3,4,1,5))  # (B, H, W, num_heads, Dk, head_dim)
        Q = Q.permute((0,2,3,4,1,5))  # (B, H, W, num_heads, Dq, head_dim)

        # Scaled dot-product attention: attn = softmax(Q @ K^T / sqrt(d_k))
        attn = torch.matmul(Q, K) * self.attn_scale  # (B, H, W, num_heads, Dq, Dk)
        attn = F.softmax(attn, dim=-1)

        # Weighted sum of values: y = attn @ V
        y = torch.matmul(attn, V)     # (B, H, W, num_heads, Dq, head_dim)
        y = y.permute((0,4,1,2,3,5))  # (B, Dq, H, W, num_heads, head_dim)
        y = y.reshape((B,Dq,H,W,C))   # (B, Dq, H, W, inner_dim) -- merge heads
        y = self.proj(y)               # (B, Dq, H, W, channels) -- project to output
        return y


class TemporalTransformer(nn.Module):
    """Full temporal transformer block: self-attention + cross-attention + MLP.

    This is Stage 2 of the forecaster architecture. It maps from Din input time
    steps to Dout output time steps using:
        1. Self-attention on the output queries (output time positions attend to each other)
        2. Cross-attention from output queries to input features (each output time
           position attends to the analysis features from all input time steps)
        3. Feed-forward MLP for channel mixing

    All three sub-layers use pre-norm and residual connections.

    In the forecaster, this is called as:
        z = temporal_transformer(pe_out, z)
    where pe_out = sinusoidal positional encoding of output time steps (the query),
    and z = analysis output with input positional encoding added (the key/value).

    Args:
        channels:    Channel dimension (same for input and output).
        mlp_dim_mul: MLP hidden dim = channels * mlp_dim_mul (default: 1).
        **kwargs:    Passed to TemporalAttention (e.g., head_dim, num_heads).
    """
    def __init__(self,
        channels,
        mlp_dim_mul=1,
        **kwargs
    ):
        super().__init__()
        self.attn1 = TemporalAttention(channels, **kwargs)  # Self-attention
        self.attn2 = TemporalAttention(channels, **kwargs)  # Cross-attention
        self.norm1 = nn.LayerNorm(channels)  # Pre-norm for self-attention
        self.norm2 = nn.LayerNorm(channels)  # Pre-norm for cross-attention
        self.norm3 = nn.LayerNorm(channels)  # Pre-norm for MLP
        self.mlp = MLP(channels, dim_mul=mlp_dim_mul)

    def forward(self, x, y):
        """Apply temporal transformer.

        Args:
            x: Query tensor (output positional encoding), shape (B, Dout, H, W, C).
            y: Key/Value tensor (analysis output + input pos enc), shape (B, Din, H, W, C).

        Returns:
            Transformed output, shape (B, Dout, H, W, C).
        """
        x = self.attn1(self.norm1(x)) + x      # Self-attention on output time queries + skip
        x = self.attn2(self.norm2(x), y) + x    # Cross-attention: output queries attend to input + skip
        return self.mlp(self.norm3(x)) + x      # Feed-forward MLP + skip


class MLP(nn.Sequential):
    """Simple 2-layer feed-forward network: Linear -> SiLU -> Linear.

    Args:
        dim:     Input and output dimension.
        dim_mul: Hidden dim = dim * dim_mul (default: 4).
    """
    def __init__(self, dim, dim_mul=4):
        inner_dim = dim * dim_mul
        sequence = [
            nn.Linear(dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, dim)
        ]
        super().__init__(*sequence)


def positional_encoding(position, dims, add_dims=()):
    """Compute sinusoidal positional encoding for time coordinates.

    Generates sin/cos encoding vectors that allow the model to distinguish
    different time positions. Used to encode both input and output time
    coordinates in the temporal transformer.

    The encoding follows the standard transformer formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/dims))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/dims))

    Args:
        position: Time coordinate tensor.
                  - 1D tensor of shape (T,) for fixed time steps
                  - 2D tensor of shape (B, T) for batch-varying time steps
        dims:     Output dimension of the encoding (must be even).
        add_dims: Tuple of dimension indices to unsqueeze (e.g., (2,3) adds H,W dims).

    Returns:
        Positional encoding tensor of shape (T, dims) or (B, T, dims),
        with additional singleton dimensions inserted at positions specified by add_dims.
    """
    # Compute the frequency scaling factors: 1/10000^(2i/dims)
    div_term = torch.exp(
        torch.arange(0, dims, 2, device=position.device) *
        (-math.log(10000.0) / dims)
    )
    # Compute sin/cos arguments: position * frequency
    if position.ndim == 1:
        arg = position[:,None] * div_term[None,:]       # (T, dims//2)
    else:
        arg = position[:,:,None] * div_term[None,None,:]  # (B, T, dims//2)

    # Concatenate sin and cos: [sin(arg), cos(arg)] along last dim
    pos_enc = torch.concat(
        [torch.sin(arg), torch.cos(arg)],
        dim=-1
    )  # Shape: (..., dims)

    # Add extra dimensions (e.g., for broadcasting to (B, T, 1, 1, dims))
    if add_dims:
        for dim in add_dims:
            pos_enc = pos_enc.unsqueeze(dim)
    return pos_enc
