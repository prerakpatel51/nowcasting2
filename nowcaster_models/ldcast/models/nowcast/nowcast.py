# Copied from LDCast paper: ldcast/models/nowcast/nowcast.py. No modifications.

"""Nowcaster (forecaster) network for precipitation nowcasting.

This module implements the AFNO-based forecaster described in Section 4.2.1 of
the paper. The forecaster takes encoded radar/satellite observations and predicts
future latent-space representations.

Architecture (3 stages):
    1. ANALYSIS:   Encode input + stack of AFNOBlock3d (global Fourier mixing)
    2. TEMPORAL:   TemporalTransformer maps Din input steps -> Dout output steps
    3. FORECAST:   Stack of AFNOBlock3d in the output time space

Multiple input support:
    The forecaster can accept multiple data sources (e.g., radar + satellite)
    simultaneously. Each gets its own autoencoder, projection, and analysis stack.
    They are fused before the forecast stage via FusionBlock3d.

Classes:
    Nowcaster              - PyTorch Lightning wrapper for training any nowcast net.
    AFNONowcastNetBasic    - Simple AFNO net using patch embed/expand (no autoencoder).
    FusionBlock3d          - Merges multiple input sources (optionally with AFNO fusion).
    AFNONowcastNetBase     - Core 3-stage forecaster (analysis -> temporal -> forecast).
    AFNONowcastNet         - AFNONowcastNetBase + autoencoder decode for pixel-space output.
"""

import collections

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from ..blocks.afno import AFNOBlock3d
from ..blocks.attention import positional_encoding, TemporalTransformer


class Nowcaster(pl.LightningModule):
    """PyTorch Lightning module for training a nowcast network.

    Wraps any nowcast_net (e.g., AFNONowcastNet) and provides training,
    validation, and test loops with MSE loss.

    Args:
        nowcast_net: The underlying nowcasting network (nn.Module).
    """
    def __init__(self, nowcast_net):
        super().__init__()
        self.nowcast_net = nowcast_net

    def forward(self, x):
        """Run the nowcast network. x depends on the specific net architecture."""
        return self.nowcast_net(x)

    def _loss(self, batch):
        """Compute MSE loss between prediction and target."""
        (x,y) = batch
        y_pred = self.forward(x)
        return (y-y_pred).square().mean()

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("train_loss", loss)
        return loss

    @torch.no_grad()
    def val_test_step(self, batch, batch_idx, split="val"):
        loss = self._loss(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log(f"{split}_loss", loss, **log_params)

    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        """AdamW optimizer with ReduceLROnPlateau scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-3,
            betas=(0.5, 0.9), weight_decay=1e-3
        )
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.25, verbose=True
        )

        optimizer_spec = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
        return optimizer_spec


class AFNONowcastNetBasic(nn.Sequential):
    """Simple AFNO nowcast net using patch embed/expand (no autoencoder).

    A basic sequential model: PatchEmbed3d -> N x AFNOBlock -> PatchExpand3d.
    This is a simpler alternative to AFNONowcastNetBase that doesn't use
    an autoencoder or temporal transformer.

    Args:
        embed_dim:  AFNO embedding dimension (default: 256).
        depth:      Number of AFNOBlock3d layers (default: 12).
        patch_size: 3D patch size for embed/expand (default: (4,4,4)).
    """
    def __init__(
        self,
        embed_dim=256,
        depth=12,
        patch_size=(4,4,4)
    ):
        patch_embed = PatchEmbed3d(
            embed_dim=embed_dim, patch_size=patch_size
        )
        blocks = nn.Sequential(
            *(AFNOBlock(embed_dim) for _ in range(depth))
        )
        patch_expand = PatchExpand3d(
            embed_dim=embed_dim, patch_size=patch_size
        )
        super().__init__(*[patch_embed, blocks, patch_expand])


class FusionBlock3d(nn.Module):
    """Fuses multiple input sources into a single representation.

    When the forecaster has multiple inputs (e.g., radar at high resolution +
    satellite at lower resolution), each input is processed independently through
    analysis, then this block combines them.

    Two fusion modes:
        1. Simple sum (afno_fusion=False): Resize all inputs to the same spatial
           resolution using transposed convolutions, then add them together.
        2. AFNO fusion (afno_fusion=True): Resize, concatenate along channels,
           then process with a Linear -> AFNOBlock3d -> Linear pipeline.

    The size_ratios parameter handles different spatial resolutions:
        - size_ratio=1: Input is already at target resolution (identity)
        - size_ratio=2: Input is 2x smaller, needs one stage of 2x upsampling
        - size_ratio=4: Input is 4x smaller, needs two stages of 2x upsampling

    Args:
        dim:          Channel dimension(s). Single int or tuple of ints per input.
        size_ratios:  Spatial size ratio of each input relative to target (e.g., (1, 2)).
        dim_out:      Output channel dimension (default: dim of first input).
        afno_fusion:  If True, use AFNO-based fusion instead of simple sum (default: False).
    """
    def __init__(self, dim, size_ratios, dim_out=None, afno_fusion=False):
        super().__init__()

        N_sources = len(size_ratios)
        if not isinstance(dim, collections.abc.Sequence):
            dim = (dim,) * N_sources
        if dim_out is None:
            dim_out = dim[0]

        # Build upsampling modules for each input source
        self.scale = nn.ModuleList()
        for (i,size_ratio) in enumerate(size_ratios):
            if size_ratio == 1:
                # Already at target resolution
                scale = nn.Identity()
            else:
                # Chain of 2x spatial upsamplings via transposed conv
                scale = []
                while size_ratio > 1:
                    scale.append(nn.ConvTranspose3d(
                        dim[i], dim_out if size_ratio==2 else dim[i],
                        kernel_size=(1,3,3), stride=(1,2,2),
                        padding=(0,1,1), output_padding=(0,1,1)
                    ))
                    size_ratio //= 2
                scale = nn.Sequential(*scale)
            self.scale.append(scale)

        self.afno_fusion = afno_fusion

        if self.afno_fusion:
            if N_sources > 1:
                # Concatenate all sources and mix with AFNO
                self.fusion = nn.Sequential(
                    nn.Linear(sum(dim), sum(dim)),
                    AFNOBlock3d(dim*N_sources, mlp_ratio=2),
                    nn.Linear(sum(dim), dim_out)
                )
            else:
                self.fusion = nn.Identity()

    def resize_proj(self, x, i):
        """Resize input i to target spatial resolution.

        Converts from channels-last to channels-first for Conv3d, then back.
        """
        x = x.permute(0,4,1,2,3)  # (B,D,H,W,C) -> (B,C,D,H,W) for ConvTranspose3d
        x = self.scale[i](x)       # Upsample spatially
        x = x.permute(0,2,3,4,1)  # (B,C,D,H',W') -> (B,D,H',W',C)
        return x

    def forward(self, x):
        """Fuse multiple input sources.

        Args:
            x: List of tensors, one per input source. Each (B, D, H_i, W_i, C_i).

        Returns:
            Fused tensor of shape (B, D, H, W, dim_out).
        """
        # Resize all inputs to target spatial resolution
        x = [self.resize_proj(xx, i) for (i, xx) in enumerate(x)]
        if self.afno_fusion:
            # Concatenate along channels and process with AFNO
            x = torch.concat(x, axis=-1)
            x = self.fusion(x)
        else:
            # Simple element-wise sum
            x = sum(x)
        return x


class AFNONowcastNetBase(nn.Module):
    """Core 3-stage AFNO forecaster (Section 4.2.1 of the paper).

    This is the main forecaster architecture that processes encoded observations
    and produces predictions in latent space. It consists of three stages:

    Stage 1 - Analysis:
        For each input source: autoencoder.encode -> 1x1 conv projection ->
        stack of AFNOBlock3d (e.g., 4 blocks). Processes the input representation
        with global Fourier mixing.

    Stage 2 - Temporal Transformer:
        Maps from Din input time steps to Dout output time steps using
        cross-attention along the time dimension. The query is constructed from
        sinusoidal positional encoding of the output time coordinates.
        Only used when input_patches != output_patches.

    Stage 3 - Forecast:
        Stack of AFNOBlock3d (e.g., 4 blocks) operating in the output time space.
        Identical architecture to the analysis stage but with its own learned weights.

    Multiple input support:
        Each input source gets its own autoencoder, projection, and analysis stack.
        After the temporal transformer, all sources are merged via FusionBlock3d
        before entering the shared forecast stage.

    Example with 2 input sources:
        Input 1 (radar):     encode -> proj -> 4x AFNO -> temporal_transformer ---+
                                                                                  +-- Fusion -> 4x AFNO forecast
        Input 2 (satellite): encode -> proj -> 4x AFNO -> temporal_transformer ---+

    Args:
        autoencoder:       Pretrained autoencoder(s). Single or list of autoencoders.
        embed_dim:         AFNO embedding dimension. Single int or list per input (default: 128).
        embed_dim_out:     Output embedding dimension (default: same as first embed_dim).
        analysis_depth:    Number of AFNO blocks in analysis. Single int or list per input (default: 4).
        forecast_depth:    Number of AFNO blocks in forecast stage (default: 4).
        input_patches:     Number of temporal patches per input. Single int or tuple (default: (1,)).
        input_size_ratios: Spatial size ratio of each input (default: (1,)).
                           E.g., (1, 2) means second input is 2x smaller spatially.
        output_patches:    Number of output time steps Dout (default: 2).
        train_autoenc:     If True, fine-tune the autoencoder during training (default: False).
        afno_fusion:       If True, use AFNO-based fusion for multiple inputs (default: False).
    """
    def __init__(
        self,
        autoencoder,
        embed_dim=128,
        embed_dim_out=None,
        analysis_depth=4,
        forecast_depth=4,
        input_patches=(1,),
        input_size_ratios=(1,),
        output_patches=2,
        train_autoenc=False,
        afno_fusion=False
    ):
        super().__init__()

        self.train_autoenc = train_autoenc

        # --- Normalize all arguments to lists (one entry per input source) ---

        # Wrap single autoencoder in a list
        if not isinstance(autoencoder, collections.abc.Sequence):
            autoencoder = [autoencoder]
        # Wrap single input_patches in a list
        if not isinstance(input_patches, collections.abc.Sequence):
            input_patches = [input_patches]
        num_inputs = len(autoencoder)  # Number of input data sources

        # Broadcast embed_dim to list of length num_inputs
        if not isinstance(embed_dim, collections.abc.Sequence):
            embed_dim = [embed_dim] * num_inputs
        # Default output dim = first input's embed_dim
        if embed_dim_out is None:
            embed_dim_out = embed_dim[0]
        # Broadcast analysis_depth to list
        if not isinstance(analysis_depth, collections.abc.Sequence):
            analysis_depth = [analysis_depth] * num_inputs

        self.embed_dim = embed_dim
        self.embed_dim_out = embed_dim_out
        self.output_patches = output_patches

        # --- Stage 1: Encoding + Analysis (one pipeline per input source) ---
        self.autoencoder = nn.ModuleList()  # Pretrained VAE encoders
        self.proj = nn.ModuleList()         # 1x1 conv: autoencoder latent -> embed_dim
        self.analysis = nn.ModuleList()     # Stacks of AFNOBlock3d
        for i in range(num_inputs):
            # Freeze or unfreeze autoencoder parameters
            ae = autoencoder[i].requires_grad_(train_autoenc)
            self.autoencoder.append(ae)

            # 1x1 Conv3d projects from autoencoder hidden channels to AFNO embed_dim
            proj = nn.Conv3d(ae.hidden_width, embed_dim[i], kernel_size=1)
            self.proj.append(proj)

            # Stack of AFNO blocks for analysis (e.g., 4 blocks)
            analysis = nn.Sequential(
                *(AFNOBlock3d(embed_dim[i]) for _ in range(analysis_depth[i]))
            )
            self.analysis.append(analysis)

        # --- Stage 2: Temporal Transformer (Din -> Dout mapping) ---
        # Only needed when any input's time steps differ from the output time steps
        self.use_temporal_transformer = \
            any((ipp != output_patches) for ipp in input_patches)
        if self.use_temporal_transformer:
            # One temporal transformer per input source
            self.temporal_transformer = nn.ModuleList(
                TemporalTransformer(embed_dim[i]) for i in range(num_inputs)
            )

        # --- Fusion: Merge multiple input sources ---
        self.fusion = FusionBlock3d(embed_dim, input_size_ratios,
            afno_fusion=afno_fusion, dim_out=embed_dim_out)

        # --- Stage 3: Forecast (operates in output time space) ---
        self.forecast = nn.Sequential(
            *(AFNOBlock3d(embed_dim_out) for _ in range(forecast_depth))
        )

    def add_pos_enc(self, x, t):
        """Add sinusoidal positional encoding of time coordinates to the features.

        If the autoencoder compressed the time dimension (e.g., from 4 steps to 2),
        the time coordinates are correspondingly downsampled via average pooling.

        Args:
            x: Feature tensor of shape (B, D, H, W, C) in channels-last format.
            t: Time coordinate tensor of shape (B, T) where T >= D.

        Returns:
            x + positional_encoding(t), same shape as x.
        """
        if t.shape[1] != x.shape[1]:
            # Autoencoder compressed time dimension: downsample time coordinates
            ds_factor = t.shape[1] // x.shape[1]
            t = F.avg_pool1d(t.unsqueeze(1), ds_factor)[:,0,:]

        pos_enc = positional_encoding(t, x.shape[-1], add_dims=(2,3))
        return x + pos_enc

    def forward(self, x):
        """Full forward pass through the 3-stage forecaster.

        Args:
            x: List of (input_tensor, time_coordinates) tuples, one per input source.
               - input_tensor: (B, C_raw, T_in, H, W) raw input data
               - time_coordinates: (B, T_in) relative time values for positional encoding

        Returns:
            Prediction tensor of shape (B, C, Dout, H, W) in channels-first format.
        """
        # Unzip list of (tensor, time) tuples
        (x, t_relative) = list(zip(*x))

        def process_input(i):
            """Process a single input source through encode -> analysis -> temporal."""
            # --- ENCODE: VAE encoder compresses raw input to latent space ---
            z = self.autoencoder[i].encode(x[i])[0]  # (B, C_latent, Din, H, W)

            # --- PROJECT: 1x1 conv maps latent channels to AFNO embedding dim ---
            z = self.proj[i](z)  # (B, embed_dim, Din, H, W)

            # --- Convert to channels-last for AFNO ---
            z = z.permute(0,2,3,4,1)  # (B, Din, H, W, embed_dim)

            # --- STAGE 1: ANALYSIS -- stack of AFNO blocks ---
            z = self.analysis[i](z)  # (B, Din, H, W, embed_dim)

            if self.use_temporal_transformer:
                # --- STAGE 2: TEMPORAL TRANSFORMER -- map Din -> Dout ---

                # Add sinusoidal positional encoding of INPUT time coordinates
                z = self.add_pos_enc(z, t_relative[i])

                # Create sinusoidal positional encoding for OUTPUT time steps (1..Dout)
                # This becomes the QUERY: "what times should I predict?"
                expand_shape = z.shape[:1] + (-1,) + z.shape[2:]
                pos_enc_output = positional_encoding(
                    torch.arange(1,self.output_patches+1, device=z.device),
                    self.embed_dim[i], add_dims=(0,2,3)
                )
                pe_out = pos_enc_output.expand(*expand_shape)  # (B, Dout, H, W, embed_dim)

                # Cross-attention: output queries attend to analysis features
                z = self.temporal_transformer[i](pe_out, z)  # (B, Dout, H, W, embed_dim)
            return z

        # Process each input source independently
        x = [process_input(i) for i in range(len(x))]

        # --- FUSION: Merge all input sources into one representation ---
        x = self.fusion(x)  # (B, Dout, H, W, embed_dim_out)

        # --- STAGE 3: FORECAST -- stack of AFNO blocks in output space ---
        x = self.forecast(x)  # (B, Dout, H, W, embed_dim_out)

        # Convert back to channels-first format
        return x.permute(0,4,1,2,3)  # (B, embed_dim_out, Dout, H, W)


class AFNONowcastNet(AFNONowcastNetBase):
    """Full nowcasting network with autoencoder decode for pixel-space output.

    Extends AFNONowcastNetBase by adding:
        1. A 1x1 Conv3d to project from embed_dim back to autoencoder hidden channels
        2. Autoencoder decode to reconstruct the final pixel-space prediction

    Used for non-generative (deterministic) prediction, where the model directly
    outputs a single forecast rather than conditioning a diffusion model.

    Pipeline:
        AFNONowcastNetBase.forward() -> 1x1 conv back-projection -> autoencoder.decode()

    Args:
        autoencoder:        Pretrained autoencoder(s) for encoding inputs.
        output_autoencoder: Autoencoder for decoding outputs (default: same as first input autoencoder).
        **kwargs:           All other arguments passed to AFNONowcastNetBase.
    """
    def __init__(self, autoencoder, output_autoencoder=None, **kwargs):
        super().__init__(autoencoder, **kwargs)
        # Default: use the first input autoencoder for decoding
        if output_autoencoder is None:
            output_autoencoder = autoencoder[0]
        self.output_autoencoder = output_autoencoder.requires_grad_(
            self.train_autoenc)
        # 1x1 Conv3d: project AFNO output back to autoencoder's latent channels
        self.out_proj = nn.Conv3d(
            self.embed_dim_out, output_autoencoder.hidden_width, kernel_size=1
        )

    def forward(self, x):
        """Full forward pass: encode -> analyze -> forecast -> decode.

        Args:
            x: List of (input_tensor, time_coordinates) tuples.

        Returns:
            Pixel-space prediction tensor of shape (B, C_out, Dout, H_full, W_full).
        """
        # Run the 3-stage forecaster (returns latent-space prediction)
        x = super().forward(x)      # (B, embed_dim_out, Dout, H, W)
        # Project back to autoencoder latent channels
        x = self.out_proj(x)        # (B, hidden_width, Dout, H, W)
        # Decode to pixel space
        return self.output_autoencoder.decode(x)  # (B, C_out, Dout, H_full, W_full)
