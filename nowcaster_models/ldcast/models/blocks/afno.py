# Copied from LDCast paper: ldcast/models/blocks/afno.py. No modifications.

"""Adaptive Fourier Neural Operator (AFNO) blocks for the ldcast forecaster.

Reference: https://github.com/NVlabs/AFNO-transformer

AFNO replaces standard self-attention with operations in the Fourier (frequency)
domain. Instead of computing O(N^2) pairwise attention, it:
  1. Transforms spatial data to frequency space via FFT  (O(N log N))
  2. Applies a learnable block-diagonal MLP to the frequency coefficients
  3. Applies hard thresholding (keep only low-frequency modes) and soft
     shrinkage (promote sparsity)
  4. Transforms back via inverse FFT

This gives global spatial mixing (every frequency mode "sees" the whole image)
at much lower computational cost than self-attention.

Classes:
    Mlp              - Simple 2-layer feed-forward network (pixelwise MLP)
    AFNO2D           - 2D Adaptive Fourier Neural Operator (spatial H x W)
    Block            - 2D AFNO block = LayerNorm + AFNO2D + skip + LayerNorm + MLP + skip
    AFNO3D           - 3D Adaptive Fourier Neural Operator (temporal+spatial D x H x W)
    AFNOBlock3d      - 3D AFNO block = LayerNorm + AFNO3D + skip + LayerNorm + MLP + skip
    PatchEmbed3d     - Converts raw input into patch embeddings via 3D convolution
    PatchExpand3d    - Reverses patch embedding back to original resolution
    AFNOCrossAttentionBlock3d - 3D AFNO block with cross-attention from a conditioning source
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention import TemporalAttention


class Mlp(nn.Module):
    """Pixelwise 2-layer feed-forward network (MLP).

    Applied independently at each spatial location (like a 1x1 convolution).
    Used inside every AFNO block for channel mixing after the Fourier filter.

    Architecture: Linear -> GELU -> Dropout -> Linear -> Dropout

    Args:
        in_features:     Input channel dimension.
        hidden_features: Hidden layer dimension (defaults to in_features).
        out_features:    Output channel dimension (defaults to in_features).
        act_layer:       Activation function class (default: nn.GELU).
        drop:            Dropout probability (default: 0.0 = no dropout).
    """
    def __init__(
        self,
        in_features, hidden_features=None, out_features=None,
        act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # Project up to hidden dim
        self.act = act_layer()                                # Non-linearity (GELU)
        self.fc2 = nn.Linear(hidden_features, out_features)  # Project back down
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        """Forward pass. x shape: (..., in_features) -> (..., out_features)."""
        x = self.fc1(x)    # (..., in_features) -> (..., hidden_features)
        x = self.act(x)    # Element-wise GELU activation
        x = self.drop(x)
        x = self.fc2(x)    # (..., hidden_features) -> (..., out_features)
        x = self.drop(x)
        return x


class AFNO2D(nn.Module):
    """2D Adaptive Fourier Neural Operator.

    Core spectral filtering layer that replaces self-attention for 2D spatial data.
    Operates in the frequency domain using FFT to achieve global mixing with O(N log N)
    complexity instead of O(N^2) for standard attention.

    How it works:
        1. Apply 2D real FFT to transform (H, W) spatial dims to frequency domain
        2. Split channels into num_blocks groups (block-diagonal structure for efficiency)
        3. Keep only the lowest-frequency modes (hard thresholding)
        4. Apply a 2-layer complex-valued MLP to the kept frequency coefficients:
           - Layer 1: complex matmul + bias + ReLU
           - Layer 2: complex matmul + bias (no activation)
           Complex multiplication is done manually: Re(AB) = Re(A)Re(B) - Im(A)Im(B)
        5. Apply soft shrinkage to promote sparsity in frequency coefficients
        6. Inverse FFT back to spatial domain
        7. Residual connection: output = filtered + original input

    Args:
        hidden_size:              Channel dimension (must be divisible by num_blocks).
        num_blocks:               Number of channel groups for block-diagonal structure (default: 8).
        sparsity_threshold:       Lambda for soft shrinkage (default: 0.01). Values with
                                  magnitude below this are zeroed out.
        hard_thresholding_fraction: Fraction of frequency modes to keep (default: 1.0 = keep all).
                                    Setting < 1.0 zeros out high-frequency modes.
        hidden_size_factor:       Expansion factor for the hidden layer in the frequency MLP (default: 1).
    """
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks  # Channels per block
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02  # Weight initialization scale

        # Learnable weights for the 2-layer complex MLP in frequency domain.
        # Shape [2, ...] stores weights for real and imaginary parts separately.
        # Layer 1: block_size -> block_size * hidden_size_factor
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        # Layer 2: block_size * hidden_size_factor -> block_size
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        """Forward pass through 2D AFNO filter.

        Args:
            x: Input tensor of shape (B, H, W, C) in channels-last format.

        Returns:
            Filtered tensor of shape (B, H, W, C) with residual connection.
        """
        bias = x  # Save input for residual connection

        dtype = x.dtype
        x = x.float()  # FFT requires float32
        B, H, W, C = x.shape

        # Step 1: 2D real FFT over spatial dimensions (H, W)
        # Output shape: (B, H, W//2+1, C) - rfft2 only returns non-redundant half
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")

        # Step 2: Split channels into num_blocks groups
        # (B, H, W//2+1, C) -> (B, H, W//2+1, num_blocks, block_size)
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        # Allocate output tensors for the 2-layer complex MLP
        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        # Step 3: Hard thresholding - determine which frequency modes to keep
        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        # Step 4a: Layer 1 of complex MLP with ReLU activation
        # Complex multiplication: Re(out) = Re(W)*Re(x) - Im(W)*Im(x) + Re(b)
        #                         Im(out) = Im(W)*Re(x) + Re(W)*Im(x) + Im(b)
        # Only applied to the kept (low-frequency) modes; rest stays zero.
        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        # Step 4b: Layer 2 of complex MLP (no activation - applied after via softshrink)
        o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        # Step 5: Recombine real and imaginary parts, apply soft shrinkage for sparsity
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)  # Zero out small coefficients
        x = torch.view_as_complex(x)  # Convert back to complex tensor

        # Step 6: Inverse FFT back to spatial domain
        x = x.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1,2), norm="ortho")
        x = x.type(dtype)  # Restore original dtype (e.g., float16)

        # Step 7: Residual connection
        return x + bias


class Block(nn.Module):
    """2D AFNO Block: LayerNorm + AFNO2D filter + skip + LayerNorm + MLP + skip.

    This is the building block used in the original FourCastNet architecture for
    2D atmospheric data. Each block consists of:
        1. LayerNorm -> AFNO2D filter (global Fourier mixing) -> residual skip
        2. LayerNorm -> Pixelwise MLP (channel mixing) -> residual skip

    The "double_skip" option adds an extra residual connection between the AFNO
    filter and the MLP, which helps gradient flow during training.

    Args:
        dim:                      Channel dimension.
        mlp_ratio:                MLP hidden dim = dim * mlp_ratio (default: 4.0).
        drop:                     Dropout probability in MLP (default: 0.0).
        drop_path:                Unused, kept for API compatibility.
        act_layer:                Activation function class (default: nn.GELU).
        norm_layer:               Normalization layer class (default: nn.LayerNorm).
        double_skip:              If True, add extra skip between AFNO and MLP (default: True).
        num_blocks:               Number of channel groups in AFNO2D (default: 8).
        sparsity_threshold:       Soft shrinkage lambda in AFNO2D (default: 0.01).
        hard_thresholding_fraction: Fraction of frequency modes to keep (default: 1.0).
    """
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)         # Pre-norm before AFNO filter
        self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction)
        self.norm2 = norm_layer(dim)         # Pre-norm before MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        """Forward pass. x shape: (B, H, W, C) -> (B, H, W, C)."""
        residual = x
        x = self.norm1(x)          # Normalize
        x = self.filter(x)         # AFNO2D: FFT -> complex MLP -> IFFT (+ internal residual)

        if self.double_skip:
            x = x + residual       # First skip connection (after AFNO filter)
            residual = x           # Reset residual for the MLP skip

        x = self.norm2(x)          # Normalize
        x = self.mlp(x)            # Pixelwise MLP for channel mixing
        x = x + residual           # Second skip connection (after MLP)
        return x



class AFNO3D(nn.Module):
    """3D Adaptive Fourier Neural Operator.

    Extends AFNO2D to 3D (temporal + spatial). Applies the Fourier filter over
    all three dimensions (D, H, W) simultaneously using 3D FFT, enabling global
    mixing across both space and time.

    Used in the ldcast forecaster where D = number of time steps, H and W are
    the spatial dimensions of the latent space.

    The architecture is identical to AFNO2D but uses rfftn/irfftn for 3D transforms.

    Args:
        hidden_size:              Channel dimension (must be divisible by num_blocks).
        num_blocks:               Number of channel groups (default: 8).
        sparsity_threshold:       Soft shrinkage lambda (default: 0.01).
        hard_thresholding_fraction: Fraction of frequency modes to keep (default: 1.0).
        hidden_size_factor:       MLP expansion factor in frequency domain (default: 1).
    """
    def __init__(
        self, hidden_size, num_blocks=8, sparsity_threshold=0.01,
        hard_thresholding_fraction=1, hidden_size_factor=1
    ):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        # Same weight structure as AFNO2D (shared across all spatial/temporal positions)
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        """Forward pass through 3D AFNO filter.

        Args:
            x: Input tensor of shape (B, D, H, W, C) in channels-last format.
               D = time steps, H = height, W = width, C = channels.

        Returns:
            Filtered tensor of shape (B, D, H, W, C) with residual connection.
        """
        bias = x  # Save for residual

        dtype = x.dtype
        x = x.float()  # FFT requires float32
        B, D, H, W, C = x.shape

        # Step 1: 3D real FFT over (D, H, W) - transforms space+time to frequency domain
        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")
        # Shape: (B, D, H, W//2+1, C) -> split into blocks
        x = x.reshape(B, D, H, W // 2 + 1, self.num_blocks, self.block_size)

        # Allocate output buffers (zeros = implicit hard thresholding for high-freq modes)
        o1_real = torch.zeros([B, D, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, D, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        # Hard thresholding: only process the lowest-frequency modes
        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        # Layer 1: Complex matmul + ReLU (only on kept low-frequency modes)
        o1_real[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        # Layer 2: Complex matmul (no activation before softshrink)
        o2_real[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        # Soft shrinkage: zero out small-magnitude frequency coefficients for sparsity
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)

        # Inverse 3D FFT back to spatial+temporal domain
        x = x.reshape(B, D, H, W // 2 + 1, C)
        x = torch.fft.irfftn(x, s=(D, H, W), dim=(1,2,3), norm="ortho")
        x = x.type(dtype)

        return x + bias  # Residual connection


class AFNOBlock3d(nn.Module):
    """3D AFNO Block: the main building block of the ldcast forecaster.

    Each block performs:
        1. LayerNorm -> AFNO3D filter (global Fourier mixing in D,H,W) -> skip
        2. LayerNorm -> Pixelwise MLP (channel mixing) -> skip

    This is stacked multiple times (e.g., analysis_depth=4) to form the
    analysis and forecast stages of the forecaster architecture.

    Args:
        dim:                      Channel dimension.
        mlp_ratio:                MLP hidden dim = dim * mlp_ratio (default: 4.0).
        drop:                     Dropout probability (default: 0.0).
        act_layer:                Activation class (default: nn.GELU).
        norm_layer:               Normalization class (default: nn.LayerNorm).
        double_skip:              Extra skip connection between AFNO and MLP (default: True).
        num_blocks:               Channel groups in AFNO3D (default: 8).
        sparsity_threshold:       Soft shrinkage lambda (default: 0.01).
        hard_thresholding_fraction: Fraction of modes to keep (default: 1.0).
        data_format:              "channels_last" (B,D,H,W,C) or "channels_first" (B,C,D,H,W).
                                  AFNO operates in channels_last internally; if channels_first,
                                  permutes automatically.
        mlp_out_features:         If set, MLP output dim differs from input dim (default: None).
    """
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            data_format="channels_last",
            mlp_out_features=None,
        ):
        super().__init__()
        self.norm_layer = norm_layer
        self.norm1 = norm_layer(dim)         # Pre-norm before AFNO3D filter
        self.filter = AFNO3D(dim, num_blocks, sparsity_threshold,
            hard_thresholding_fraction)      # Core Fourier operator
        self.norm2 = norm_layer(dim)         # Pre-norm before MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, out_features=mlp_out_features,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop
        )
        self.double_skip = double_skip
        self.channels_first = (data_format == "channels_first")

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, D, H, W, C) if channels_last, or (B, C, D, H, W) if channels_first.

        Returns:
            Output tensor with same shape as input.
        """
        if self.channels_first:
            # AFNO natively uses channels-last; convert from (B,C,D,H,W) -> (B,D,H,W,C)
            x = x.permute(0,2,3,4,1)

        residual = x
        x = self.norm1(x)          # LayerNorm over channel dim
        x = self.filter(x)         # AFNO3D: 3D FFT -> complex MLP -> 3D IFFT + residual

        if self.double_skip:
            x = x + residual       # First skip (after AFNO filter)
            residual = x           # Update residual for MLP

        x = self.norm2(x)          # LayerNorm
        x = self.mlp(x)            # Pixelwise MLP: channel mixing at each (d,h,w) location
        x = x + residual           # Second skip (after MLP)

        if self.channels_first:
            # Convert back to (B,C,D,H,W)
            x = x.permute(0,4,1,2,3)

        return x


class PatchEmbed3d(nn.Module):
    """3D Patch Embedding: converts raw input volume into patch tokens.

    Splits the input into non-overlapping 3D patches and projects each patch
    to an embedding vector using a strided 3D convolution.

    For example, with patch_size=(4,4,4) and embed_dim=256:
        Input:  (B, 1, 20, 64, 64) -> patches of 4x4x4
        Output: (B, 5, 16, 16, 256) in channels-last format

    Args:
        patch_size: (D_patch, H_patch, W_patch) size of each 3D patch (default: (4,4,4)).
        in_chans:   Number of input channels (default: 1).
        embed_dim:  Embedding dimension per patch (default: 256).
    """
    def __init__(self, patch_size=(4,4,4), in_chans=1, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        # Strided conv with kernel=stride=patch_size: non-overlapping patches
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """Embed patches. x: (B, C_in, D, H, W) -> (B, D', H', W', embed_dim)."""
        x = self.proj(x)               # (B, embed_dim, D', H', W')
        x = x.permute(0,2,3,4,1)       # Convert to channels-last: (B, D', H', W', embed_dim)
        return x


class PatchExpand3d(nn.Module):
    """3D Patch Expansion: reverses patch embedding back to original resolution.

    Takes patch tokens and expands them back to the full spatial/temporal resolution
    by predicting all voxels within each patch from the embedding vector.

    For example, with patch_size=(4,4,4) and embed_dim=256:
        Input:  (B, 5, 16, 16, 256)
        Output: (B, 1, 20, 64, 64) in channels-first format

    Args:
        patch_size: (D_patch, H_patch, W_patch) size of each 3D patch (default: (4,4,4)).
        out_chans:  Number of output channels (default: 1).
        embed_dim:  Input embedding dimension per patch (default: 256).
    """
    def __init__(self, patch_size=(4,4,4), out_chans=1, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        # Project embedding to all voxels in the patch: embed_dim -> out_chans * prod(patch_size)
        self.proj = nn.Linear(embed_dim, out_chans*np.prod(patch_size))

    def forward(self, x):
        """Expand patches. x: (B, D', H', W', embed_dim) -> (B, C_out, D, H, W)."""
        x = self.proj(x)  # (B, D', H', W', out_chans * p0 * p1 * p2)
        # Rearrange: unpack the flattened patch voxels back to spatial dimensions
        x = rearrange(
            x,
            "b d h w (p0 p1 p2 c_out) -> b c_out (d p0) (h p1) (w p2)",
            p0=self.patch_size[0],
            p1=self.patch_size[1],
            p2=self.patch_size[2],
            d=x.shape[1],
            h=x.shape[2],
            w=x.shape[3],
        )
        return x  # (B, C_out, D, H, W) in channels-first format


class AFNOCrossAttentionBlock3d(nn.Module):
    """3D AFNO Block with cross-attention from a conditioning source.

    Used in the diffusion UNet to inject conditioning information from the
    forecaster into the denoising process. Instead of standard cross-attention,
    it concatenates the UNet features (x) with the conditioning features (y)
    along the channel dimension and processes them jointly through AFNO.

    Architecture:
        1. Concatenate: [LayerNorm(x), y] along channels -> (dim + context_dim) channels
        2. Linear projection + residual on the concatenated tensor
        3. AFNO3D filter on the concatenated tensor (global Fourier mixing) + residual
        4. MLP: projects back from (dim + context_dim) -> dim, with skip from x

    This is how the forecaster's predictions condition the diffusion model:
        - x = UNet feature map at a given resolution
        - y = corresponding forecaster cascade output at the same resolution

    Args:
        dim:                      Channel dimension of the UNet features (x).
        context_dim:              Channel dimension of the conditioning features (y).
        mlp_ratio:                MLP hidden dim expansion (default: 2.0).
        drop:                     Dropout probability (default: 0.0).
        act_layer:                Activation class (default: nn.GELU).
        norm_layer:               Normalization class (default: nn.Identity).
        double_skip:              Unused, kept for API compatibility.
        num_blocks:               Channel groups in AFNO3D (default: 8).
        sparsity_threshold:       Soft shrinkage lambda (default: 0.01).
        hard_thresholding_fraction: Fraction of modes to keep (default: 1.0).
        data_format:              "channels_last" or "channels_first".
        timesteps:                Unused, kept for API compatibility.
    """
    def __init__(
        self,
        dim,
        context_dim,
        mlp_ratio=2.,
        drop=0.,
        act_layer=nn.GELU,
        norm_layer=nn.Identity,
        double_skip=True,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        data_format="channels_last",
        timesteps=None
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)                        # Normalize UNet features
        self.norm2 = norm_layer(dim+context_dim)            # Normalize after concat
        mlp_hidden_dim = int((dim+context_dim) * mlp_ratio)
        self.pre_proj = nn.Linear(dim+context_dim, dim+context_dim)  # Mixing projection
        self.filter = AFNO3D(dim+context_dim, num_blocks, sparsity_threshold,
            hard_thresholding_fraction)                     # AFNO on joint channels
        self.mlp = Mlp(
            in_features=dim+context_dim,
            out_features=dim,                               # Projects back to UNet dim
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop
        )
        self.channels_first = (data_format == "channels_first")

    def forward(self, x, y):
        """Forward pass with cross-attention conditioning.

        Args:
            x: UNet features, shape (B, C, D, H, W) or (B, D, H, W, C).
            y: Conditioning from forecaster, shape (B, C_ctx, D, H, W) or (B, D, H, W, C_ctx).

        Returns:
            Conditioned output, same shape as x.
        """
        if self.channels_first:
            # Convert to channels-last for AFNO
            x = x.permute(0,2,3,4,1)  # (B,C,D,H,W) -> (B,D,H,W,C)
            y = y.permute(0,2,3,4,1)  # (B,C_ctx,D,H,W) -> (B,D,H,W,C_ctx)

        # Step 1: Concatenate normalized UNet features with conditioning
        xy = torch.concat((self.norm1(x),y), axis=-1)  # (B,D,H,W, dim+context_dim)
        # Step 2: Linear projection with residual
        xy = self.pre_proj(xy) + xy
        # Step 3: AFNO3D filter (global Fourier mixing on joint channels) with residual
        xy = self.filter(self.norm2(xy)) + xy
        # Step 4: MLP projects back to dim, with skip from original x
        x = self.mlp(xy) + x

        if self.channels_first:
            x = x.permute(0,4,1,2,3)  # Back to (B,C,D,H,W)

        return x
