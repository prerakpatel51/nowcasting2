# Copied from LDCast paper: ldcast/models/blocks/resnet.py. No modifications.

"""3D Residual block for the ldcast model.

Used in the autoencoder (encoder/decoder) and in the AFNONowcastNetCascade
for progressive downsampling of the conditioning signal.

Classes:
    ResBlock3D - 3D residual block with optional up/downsampling and spectral normalization.
"""

from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as sn

from ..utils import activation, normalization


class ResBlock3D(nn.Module):
    """3D Residual Block with optional resampling.

    Standard residual block pattern:
        shortcut: x -> [resample] -> [1x1 proj if channels change] -> x_in
        main:     x -> norm -> act -> conv1 -> norm -> act -> conv2
        output:   main + shortcut

    Supports:
        - Downsampling via strided convolution (resample="down")
        - Upsampling via transposed convolution (resample="up")
        - Channel dimension changes via 1x1 convolution on the shortcut
        - Spectral normalization for training stability (used in discriminator/GAN)

    Args:
        in_channels:     Number of input channels.
        out_channels:    Number of output channels (can differ from in_channels).
        resample:        None (no resampling), "down" (strided conv), or "up" (transposed conv).
        resample_factor: Stride/scale factor for resampling, e.g. (1,2,2) for spatial-only 2x.
        kernel_size:     Convolution kernel size (default: (3,3,3)).
        act:             Activation function name(s). Can be a single string (used for both)
                         or a tuple of two strings (default: 'swish').
        norm:            Normalization type: 'group', 'batch', 'identity', or None (default: 'group').
        norm_kwargs:     Extra keyword arguments for normalization layer.
        spectral_norm:   If True, apply spectral normalization to all convolutions (default: False).
    """
    def __init__(
        self, in_channels, out_channels, resample=None,
        resample_factor=(1,1,1), kernel_size=(3,3,3),
        act='swish', norm='group', norm_kwargs=None,
        spectral_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Shortcut path: 1x1 conv to match channel dims, or identity if same
        if in_channels != out_channels:
            self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

        padding = tuple(k//2 for k in kernel_size)  # Same-padding for each kernel dim

        if resample == "down":
            # Downsampling: avg_pool on shortcut, strided conv1 on main path
            self.resample = nn.AvgPool3d(resample_factor, ceil_mode=True)
            self.conv1 = nn.Conv3d(in_channels, out_channels,
                kernel_size=kernel_size, stride=resample_factor, padding=padding)
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                kernel_size=kernel_size, padding=padding)
        elif resample == "up":
            # Upsampling: trilinear interpolation on shortcut, transposed conv on main
            self.resample = nn.Upsample(
                scale_factor=resample_factor, mode='trilinear')
            self.conv1 = nn.ConvTranspose3d(in_channels, out_channels,
                kernel_size=kernel_size, padding=padding)
            # Compute output_padding for transposed conv to match exact output size
            output_padding = tuple(
                2*p+s-k for (p,s,k) in zip(padding,resample_factor,kernel_size)
            )
            self.conv2 = nn.ConvTranspose3d(out_channels, out_channels,
                kernel_size=kernel_size, stride=resample_factor,
                padding=padding, output_padding=output_padding)
        else:
            # No resampling: standard convolutions
            self.resample = nn.Identity()
            self.conv1 = nn.Conv3d(in_channels, out_channels,
                kernel_size=kernel_size, padding=padding)
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                kernel_size=kernel_size, padding=padding)

        # Activation functions (can be different for each layer)
        if isinstance(act, str):
            act = (act, act)
        self.act1 = activation(act_type=act[0])
        self.act2 = activation(act_type=act[1])

        # Normalization layers
        if norm_kwargs is None:
            norm_kwargs = {}
        self.norm1 = normalization(in_channels, norm_type=norm, **norm_kwargs)
        self.norm2 = normalization(out_channels, norm_type=norm, **norm_kwargs)

        # Optional spectral normalization (wraps conv layers)
        if spectral_norm:
            self.conv1 = sn(self.conv1)
            self.conv2 = sn(self.conv2)
            if not isinstance(self.proj, nn.Identity):
                self.proj = sn(self.proj)


    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C_in, D, H, W).

        Returns:
            Output tensor of shape (B, C_out, D', H', W') where D', H', W'
            depend on the resample setting.
        """
        # Shortcut path: project channels + resample
        x_in = self.resample(self.proj(x))
        # Main path: norm -> act -> conv -> norm -> act -> conv
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)   # Also resamples if resample="down"/"up"
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        # Residual connection
        return x + x_in
