"""Training entry point for the pixel-space LDCast diffusion model on IMERG data.

This script wires together:
    1. The IMERG data pipeline (dataloader.py -> DataModule)
    2. IdentityAutoencoder (no-op, since data is already 64x64x1)
    3. AFNONowcastNetCascade (analysis/conditioning network)
    4. UNetModel (3D UNet denoiser)
    5. LatentDiffusion (training wrapper with EMA, noise scheduling, etc.)

Usage:
    python -m nowcaster_models.ldcast.train
    python -m nowcaster_models.ldcast.train --config path/to/config.yaml
    python -m nowcaster_models.ldcast.train --lr 5e-5 --batch_size 16
"""

import argparse
import os

import torch
from omegaconf import OmegaConf

from .dataloader import setup_data
from .models.autoenc import IdentityAutoencoder
from .models.genforecast.analysis import AFNONowcastNetCascade
from .models.genforecast.unet import UNetModel
from .models.genforecast.training import setup_genforecast_training


def build_model(config):
    """Build the full diffusion model from config.

    Returns:
        (ldm, trainer): LatentDiffusion module and PyTorch Lightning Trainer.
    """
    input_timesteps = config.pipeline.input_timesteps   # 16
    output_timesteps = config.pipeline.output_timesteps  # 12

    # Model hyperparameters (can be overridden via config)
    model_cfg = config.get("model", {})
    embed_dim = model_cfg.get("embed_dim", 128)
    analysis_depth = model_cfg.get("analysis_depth", 4)
    forecast_depth = model_cfg.get("forecast_depth", 4)
    cascade_depth = model_cfg.get("cascade_depth", 3)
    model_channels = model_cfg.get("model_channels", 256)
    channel_mult = tuple(model_cfg.get("channel_mult", [1, 2, 4]))
    attention_resolutions = tuple(model_cfg.get("attention_resolutions", [1, 2]))
    num_heads = model_cfg.get("num_heads", 8)
    num_res_blocks = model_cfg.get("num_res_blocks", 2)
    lr = model_cfg.get("lr", 1e-4)
    lr_warmup = model_cfg.get("lr_warmup", 0)
    model_dir = model_cfg.get("model_dir",
        os.path.join(os.path.dirname(__file__), "checkpoints_improved_v2"))

    # New improvement flags (backward compatible defaults)
    use_groupnorm = model_cfg.get("use_groupnorm", False)
    dropout = model_cfg.get("dropout", 0.0)
    beta_schedule = model_cfg.get("beta_schedule", "linear")

    os.makedirs(model_dir, exist_ok=True)

    # Set GroupNorm flag before constructing the UNet
    from .models.diffusion.utils import set_use_groupnorm
    set_use_groupnorm(use_groupnorm)

    # Identity autoencoder (no-op for pixel-space diffusion)
    autoencoder = IdentityAutoencoder()

    # Analysis/conditioning network: AFNO forecaster with multi-scale cascade
    context_encoder = AFNONowcastNetCascade(
        [autoencoder],
        input_patches=[input_timesteps],
        input_size_ratios=[1],
        output_patches=output_timesteps,
        cascade_depth=cascade_depth,
        embed_dim=[embed_dim],
        analysis_depth=[analysis_depth],
        forecast_depth=forecast_depth,
    )

    # 3D UNet denoiser conditioned on the cascade
    unet = UNetModel(
        in_channels=1,
        out_channels=1,
        model_channels=model_channels,
        channel_mult=channel_mult,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_res_blocks=num_res_blocks,
        dims=3,
        num_timesteps=output_timesteps,
        context_ch=context_encoder.cascade_dims,
        dropout=dropout,
    )

    # Wire everything into the LatentDiffusion training wrapper
    (ldm, trainer) = setup_genforecast_training(
        unet, autoencoder, context_encoder, model_dir, lr=lr,
        lr_warmup=lr_warmup, beta_schedule=beta_schedule,
    )

    return (ldm, trainer)


def main():
    parser = argparse.ArgumentParser(description="Train LDCast diffusion model on IMERG")
    parser.add_argument("--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config_improved_v2.yaml"),
        help="Path to config YAML file")
    parser.add_argument("--lr", type=float, default=None,
        help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None,
        help="Override batch size")
    parser.add_argument("--resume", type=str, default=None,
        help="Path to checkpoint to resume training from (e.g. last.ckpt)")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    # Apply CLI overrides
    if args.lr is not None:
        config = OmegaConf.merge(config, {"model": {"lr": args.lr}})
    if args.batch_size is not None:
        config.sampling.batch_size = args.batch_size

    # Setup data
    print("Setting up data pipeline...")
    datamodule = setup_data(config)

    # Build model
    print("Building model...")
    (ldm, trainer) = build_model(config)

    # Count parameters
    total_params = sum(p.numel() for p in ldm.parameters())
    trainable_params = sum(p.numel() for p in ldm.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train
    print("Starting training...")
    trainer.fit(ldm, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
