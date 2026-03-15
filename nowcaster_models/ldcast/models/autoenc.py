"""Identity autoencoder for pixel-space diffusion.

Since our IMERG data is already small (64x64, 1 channel), there is no need
for a variational autoencoder to compress it. This module provides an
IdentityAutoencoder that acts as a no-op, allowing the rest of the LDCast
architecture (AFNONowcastNetBase, LatentDiffusion) to work unchanged.

The paper's autoencoder compressed 128x128 images to 32x32 latent space with
32 channels. Here, encode/decode are identity operations and hidden_width=1
(matching our single-channel IMERG data).
"""

import torch
from torch import nn


class IdentityAutoencoder(nn.Module):
    """No-op autoencoder that passes data through unchanged.

    Provides the same interface as the paper's VAE autoencoder:
        - encode(x) returns (x, None) to match the (sample, kl_loss) convention
        - decode(x) returns x unchanged
        - hidden_width = 1 (single channel, matching IMERG data)

    This allows all paper code to work without modification:
        - nowcast.py: z = self.autoencoder[i].encode(x[i])[0]  -> returns x[i]
        - nowcast.py: nn.Conv3d(ae.hidden_width, embed_dim, 1)  -> Conv3d(1, 128, 1)
        - diffusion.py: y = self.autoencoder.encode(y)[0]  -> returns y
        - diffusion.py: autoencoder.requires_grad_(False)  -> works (nn.Module)
    """

    def __init__(self):
        super().__init__()
        self.hidden_width = 1

    def encode(self, x):
        """Identity encode. Returns (x, None) to match VAE (sample, kl) convention."""
        return (x, None)

    def decode(self, x):
        """Identity decode. Returns x unchanged."""
        return x
