"""Inference script for the pixel-space LDCast diffusion model on IMERG data.

Loads trained weights, runs PLMS sampling to generate ensemble forecasts,
and applies inverse normalization to produce precipitation predictions.

Usage:
    python -m nowcaster_models.ldcast.forecast --checkpoint path/to/checkpoint.ckpt
    python -m nowcaster_models.ldcast.forecast --checkpoint ckpt.ckpt --num_samples 20 --steps 50
"""

import argparse
import os

import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from omegaconf import OmegaConf

from .train import build_model
from .models.diffusion.plms import PLMSSampler


def inverse_normalize(x, mean, std, threshold, fill_value):
    """Inverse of the normalize_threshold transform.

    Converts normalized model output back to precipitation in mm/hr.
    Steps (inverse of normalize_threshold):
        1. Un-standardize: x = x * std + mean
        2. Un-log: x = 10^x
        3. Un-threshold: values below original threshold -> 0

    Args:
        x: Normalized tensor or array.
        mean: Normalization mean (in log10 space).
        std: Normalization std (in log10 space).
        threshold: Original threshold value (mm/hr).
        fill_value: Original fill value for sub-threshold pixels.

    Returns:
        Precipitation values in mm/hr.
    """
    x = x * std + mean              # Un-standardize
    x = 10.0 ** x                   # Un-log10
    x[x < threshold] = 0.0          # Un-threshold: below 0.1 mm/hr -> zero
    return x


@torch.no_grad()
def generate_forecast(ldm, input_batch, num_samples=1, plms_steps=50,
                      verbose=True, clip_denoised=False, clip_range=(-5.0, 4.0)):
    """Generate ensemble forecasts using PLMS sampling.

    Args:
        ldm: Trained LatentDiffusion model.
        input_batch: Tuple (pred_batch, target_batch) from the dataloader.
            pred_batch = [(data, t_relative)] where data=(B,1,16,64,64)
        num_samples: Number of ensemble members to generate per input.
        plms_steps: Number of PLMS denoising steps (default: 50).
        verbose: Print progress info.
        clip_denoised: If True, clip predicted x0 during PLMS sampling.
        clip_range: (lo, hi) bounds for clipping in normalized space.

    Returns:
        samples: Tensor of shape (num_samples, B, 1, 12, 64, 64) in normalized space.
    """
    (pred_batch, _) = input_batch
    (data, t_relative) = pred_batch[0]

    device = next(ldm.parameters()).device
    data = data.to(device)
    t_relative = t_relative.to(device)

    B = data.shape[0]
    output_shape = (1, 12, 64, 64)  # (C, T_out, H, W)

    # Pass raw input as conditioning — apply_model calls context_encoder internally
    conditioning = [(data, t_relative)]

    # Setup PLMS sampler
    sampler = PLMSSampler(ldm)
    sampler.clip_denoised = clip_denoised
    sampler.clip_range = clip_range

    all_samples = []
    for s in range(num_samples):
        if verbose:
            print(f"Generating sample {s+1}/{num_samples}...")

        samples, _ = sampler.sample(
            S=plms_steps,
            batch_size=B,
            shape=output_shape,
            conditioning=conditioning,
            verbose=False,
            progbar=verbose,
        )
        all_samples.append(samples.cpu())

    return torch.stack(all_samples, dim=0)  # (num_samples, B, 1, 12, 64, 64)


def _render_frame(data, title_text, cmap, norm, figsize=(5, 4)):
    """Render a single frame to an RGB numpy array."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title_text)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Precipitation (mm/hr)")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def _render_compare_frame(pred_data, tgt_data, title_text, pred_label,
                          cmap, norm):
    """Render a side-by-side prediction vs target frame to RGB array."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4),
                             gridspec_kw={"right": 0.88})
    im0 = axes[0].imshow(pred_data, cmap=cmap, norm=norm, interpolation="nearest")
    axes[0].set_title(pred_label)
    im1 = axes[1].imshow(tgt_data, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("Target")
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])
    fig.colorbar(im1, cax=cbar_ax, label="Precipitation (mm/hr)")
    fig.suptitle(title_text)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def _render_three_panel_frame(input_data, pred_data, tgt_data, title_text,
                              input_label, pred_label, tgt_label, cmap, norm):
    """Render Input | Prediction | Target frame to RGB array.

    Any panel can be None to show a blank (gray) panel.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4),
                             gridspec_kw={"right": 0.90})
    for ax, data, label in zip(axes, [input_data, pred_data, tgt_data],
                                [input_label, pred_label, tgt_label]):
        if data is not None:
            ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")
        else:
            ax.set_facecolor("#e0e0e0")
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

    # Dummy image for colorbar (use the norm range)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    fig.colorbar(sm, cax=cbar_ax, label="Precipitation (mm/hr)")
    fig.suptitle(title_text, fontsize=13)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def save_forecast_gif(predictions, targets, output_dir, interval_minutes=30,
                      vmin=0, vmax=10, batch_idx=0, inputs=None):
    """Save prediction and target frames as animated GIFs.

    Args:
        predictions: (num_samples, B, 1, T_out, H, W) precipitation in mm/hr.
        targets: (B, 1, T_out, H, W) precipitation in mm/hr.
        output_dir: Directory to save GIF files.
        interval_minutes: Time interval between frames.
        vmin: Colorbar minimum.
        vmax: Colorbar maximum.
        batch_idx: Batch index label for filenames.
        inputs: (B, 1, T_in, H, W) precipitation in mm/hr, or None.
    """
    os.makedirs(output_dir, exist_ok=True)

    cmap = plt.cm.turbo.copy()
    cmap.set_under("white")
    norm = mcolors.Normalize(vmin=0.1, vmax=vmax)

    B = targets.shape[0]
    num_samples = predictions.shape[0]
    T_out = targets.shape[2]  # (B, C, T_out, H, W)
    T_in = inputs.shape[2] if inputs is not None else 0
    frame_duration_ms = 500  # 2 seconds per frame

    for b in range(B):
        target_frames = targets[b, 0]  # (T_out, H, W)

        # --- Target GIF ---
        imgs = []
        for t in range(T_out):
            img = _render_frame(target_frames[t],
                                f"Target  t=+{(t+1)*interval_minutes} min",
                                cmap, norm)
            imgs.append(img)
        gif_path = os.path.join(output_dir,
                                f"batch{batch_idx}_sample{b}_target.gif")
        imageio.mimsave(gif_path, imgs, duration=frame_duration_ms, loop=0)
        print(f"  Saved {gif_path}")

        # --- Prediction vs Target GIF per ensemble member ---
        for s in range(num_samples):
            pred_frames = predictions[s, b, 0]  # (T_out, H, W)
            imgs = []
            for t in range(T_out):
                img = _render_compare_frame(
                    pred_frames[t], target_frames[t],
                    f"t=+{(t+1)*interval_minutes} min",
                    f"Prediction (member {s+1})", cmap, norm)
                imgs.append(img)
            gif_path = os.path.join(
                output_dir,
                f"batch{batch_idx}_sample{b}_member{s+1}.gif")
            imageio.mimsave(gif_path, imgs, duration=frame_duration_ms, loop=0)
            print(f"  Saved {gif_path}")

        # --- Combined Input → Prediction → Target GIF per ensemble member ---
        if inputs is not None:
            input_frames = inputs[b, 0]  # (T_in, H, W)
            for s in range(num_samples):
                pred_frames = predictions[s, b, 0]
                imgs = []

                # Phase 1: Input window (Prediction & Target blank)
                for t in range(T_in):
                    t_min = (t - T_in + 1) * interval_minutes
                    img = _render_three_panel_frame(
                        input_frames[t], None, None,
                        f"Input phase: t={t_min} min",
                        f"Input (t={t_min} min)", "Prediction", "Target",
                        cmap, norm)
                    imgs.append(img)

                # Phase 2: Forecast (Input frozen on last frame)
                last_input = input_frames[-1]
                for t in range(T_out):
                    t_min = (t + 1) * interval_minutes
                    img = _render_three_panel_frame(
                        last_input, pred_frames[t], target_frames[t],
                        f"Forecast phase: t=+{t_min} min",
                        "Input (last)", f"Prediction (member {s+1})", "Target",
                        cmap, norm)
                    imgs.append(img)

                gif_path = os.path.join(
                    output_dir,
                    f"batch{batch_idx}_sample{b}_combined_member{s+1}.gif")
                imageio.mimsave(gif_path, imgs, duration=frame_duration_ms, loop=0)
                print(f"  Saved {gif_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate IMERG forecasts with trained LDCast model")
    parser.add_argument("--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint (.ckpt)")
    parser.add_argument("--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "/home1/ppatel2025/nowcasting2/nowcaster_models/ldcast/config_improved.yaml"),
        help="Path to config YAML file")
    parser.add_argument("--num_samples", type=int, default=1,
        help="Number of ensemble members to generate")
    parser.add_argument("--steps", type=int, default=100,
        help="Number of PLMS sampling steps")
    parser.add_argument("--output", type=str, default="forecast_output.npz",
        help="Output file path (.npz)")
    parser.add_argument("--split", type=str, default="test",
        choices=["train", "valid", "test"],
        help="Which data split to run inference on")
    parser.add_argument("--num_batches", type=int, default=3,
        help="Number of batches to process")
    parser.add_argument("--gif_dir", type=str, default="forecast_gifs_improved_v3_model",
        help="Directory to save forecast GIFs")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    # Build model architecture
    print("Building model...")
    from .models.diffusion.diffusion import LatentDiffusion
    (ldm, _) = build_model(config)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ldm.load_state_dict(checkpoint["state_dict"])
    ldm.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ldm = ldm.to(device)

    # Setup data
    print("Setting up data pipeline...")
    from .dataloader import setup_data
    datamodule = setup_data(config)
    datamodule.setup(stage="test")

    # Get dataloader for the chosen split
    if args.split == "test":
        dataloader = datamodule.test_dataloader()
    elif args.split == "valid":
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.train_dataloader()

    # Read clip settings from config (backward compatible)
    model_cfg = config.get("model", {})
    clip_denoised = model_cfg.get("clip_denoised", False)
    clip_range = tuple(model_cfg.get("clip_range", [-5.0, 4.0]))

    # Run inference
    all_predictions = []
    all_targets = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break

        print(f"\nBatch {batch_idx + 1}/{args.num_batches}")
        (pred_batch, target) = batch

        # Generate ensemble forecasts (in normalized space)
        samples = generate_forecast(
            ldm, batch,
            num_samples=args.num_samples,
            plms_steps=args.steps,
            clip_denoised=clip_denoised,
            clip_range=clip_range,
        )

        # Debug: check raw model output stats
        raw = samples.numpy()
        raw_tgt = target.numpy()
        raw_inp = pred_batch[0][0].numpy()  # (B, 1, T_in, H, W)
        print(f"  Raw predictions — min: {raw.min():.4f}, max: {raw.max():.4f}, "
              f"mean: {raw.mean():.4f}, std: {raw.std():.4f}")
        print(f"  Raw targets     — min: {raw_tgt.min():.4f}, max: {raw_tgt.max():.4f}, "
              f"mean: {raw_tgt.mean():.4f}, std: {raw_tgt.std():.4f}")

        # Inverse normalize to get precipitation in mm/hr
        inv_kwargs = dict(
            mean=config.transform.mean,
            std=config.transform.std,
            threshold=config.transform.threshold,
            fill_value=config.transform.fill_value,
        )
        samples_precip = inverse_normalize(raw.copy(), **inv_kwargs)
        target_precip = inverse_normalize(raw_tgt.copy(), **inv_kwargs)
        input_precip = inverse_normalize(raw_inp.copy(), **inv_kwargs)

        print(f"  After inv-norm preds — min: {samples_precip.min():.4f}, "
              f"max: {samples_precip.max():.4f}, nonzero: {(samples_precip > 0).sum()}")
        print(f"  After inv-norm tgts  — min: {target_precip.min():.4f}, "
              f"max: {target_precip.max():.4f}, nonzero: {(target_precip > 0).sum()}")

        all_predictions.append(samples_precip)
        all_targets.append(target_precip)

        # Save GIFs for this batch
        print(f"  Saving GIFs to {args.gif_dir}/...")
        save_forecast_gif(
            samples_precip, target_precip,
            output_dir=args.gif_dir,
            interval_minutes=config.pipeline.interval_minutes,
            batch_idx=batch_idx,
            inputs=input_precip,
        )

    # Save results
    predictions = np.concatenate(all_predictions, axis=1)  # (num_samples, N, 1, 12, 64, 64)
    targets = np.concatenate(all_targets, axis=0)           # (N, 1, 12, 64, 64)

    print(f"\nSaving to {args.output}...")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Targets shape: {targets.shape}")
    np.savez_compressed(args.output, predictions=predictions, targets=targets)
    print("Done.")


if __name__ == "__main__":
    main()
