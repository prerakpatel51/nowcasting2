"""Alpha-calibration script for validation predictions.

Two modes:
    1. predict  — run inference on the validation set, save predictions + ground truth
    2. optimize — sweep alpha to minimize MSE, plot curve, save best-alpha predictions + GIFs

Usage:
    python -m nowcaster_models.ldcast.val_alpha_calibration predict \
        --config nowcaster_models/ldcast/config_improved_v3.yaml \
        --checkpoint checkpoints_cp/best-epoch=73-val_loss_ema=0.0346.ckpt \
        --num_batches 10 --num_samples 5 --steps 50

    python -m nowcaster_models.ldcast.val_alpha_calibration optimize \
        --pred_file val_pred.npy --alpha_min 0.01 --alpha_max 3.0 --alpha_steps 300 \
        --num_gifs 5
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
from .forecast import generate_forecast, inverse_normalize, save_forecast_gif


# ──────────────────────────────────────────────────────────────────────
# Predict mode
# ──────────────────────────────────────────────────────────────────────

def run_predict(args):
    """Run model inference on the validation split and save predictions."""
    config = OmegaConf.load(args.config)

    # Build model
    print("Building model...")
    (ldm, _) = build_model(config)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ldm.load_state_dict(checkpoint["state_dict"])
    ldm.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ldm = ldm.to(device)
    print(f"Model on {device}")

    # Setup data
    print("Setting up data pipeline...")
    from .dataloader import setup_data
    datamodule = setup_data(config)
    datamodule.setup(stage="test")
    val_loader = datamodule.val_dataloader()

    # Inverse-normalize kwargs
    inv_kwargs = dict(
        mean=config.transform.mean,
        std=config.transform.std,
        threshold=config.transform.threshold,
        fill_value=config.transform.fill_value,
    )

    # Read clip settings from config
    model_cfg = config.get("model", {})
    clip_denoised = model_cfg.get("clip_denoised", False)
    clip_range = tuple(model_cfg.get("clip_range", [-5.0, 4.0]))

    all_predictions = []
    all_targets = []

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= args.num_batches:
            break

        print(f"\nBatch {batch_idx + 1}/{args.num_batches}")
        (pred_batch, target) = batch

        # Generate forecasts (normalized space)
        samples = generate_forecast(
            ldm, batch,
            num_samples=args.num_samples,
            plms_steps=args.steps,
            clip_denoised=clip_denoised,
            clip_range=clip_range,
        )
        # samples: (num_samples, B, 1, 12, 64, 64)

        raw_samples = samples.numpy()
        raw_target = target.numpy()

        print(f"  Raw preds  — min: {raw_samples.min():.4f}, max: {raw_samples.max():.4f}, "
              f"mean: {raw_samples.mean():.4f}")
        print(f"  Raw target — min: {raw_target.min():.4f}, max: {raw_target.max():.4f}, "
              f"mean: {raw_target.mean():.4f}")

        # Inverse normalize to mm/hr
        samples_precip = inverse_normalize(raw_samples.copy(), **inv_kwargs)
        target_precip = inverse_normalize(raw_target.copy(), **inv_kwargs)

        # Ensemble mean → (B, 1, 12, 64, 64)
        pred_mean = samples_precip.mean(axis=0)

        print(f"  Precip preds (ens mean) — min: {pred_mean.min():.4f}, "
              f"max: {pred_mean.max():.4f}, nonzero: {(pred_mean > 0).sum()}")
        print(f"  Precip target           — min: {target_precip.min():.4f}, "
              f"max: {target_precip.max():.4f}, nonzero: {(target_precip > 0).sum()}")

        all_predictions.append(pred_mean)
        all_targets.append(target_precip)

    # Concatenate across batches → (N, 1, 12, 64, 64)
    predictions = np.concatenate(all_predictions, axis=0)
    ground_truth = np.concatenate(all_targets, axis=0)

    print(f"\nFinal predictions shape:  {predictions.shape}")
    print(f"Final ground_truth shape: {ground_truth.shape}")

    # Save
    np.save(args.output, {"predictions": predictions, "ground_truth": ground_truth})
    print(f"Saved to {args.output}")


# ──────────────────────────────────────────────────────────────────────
# Optimize mode
# ──────────────────────────────────────────────────────────────────────

def run_optimize(args):
    """Load predictions, sweep alpha, plot MSE curve, save best-alpha results + GIFs."""
    print(f"Loading predictions from {args.pred_file}...")
    data = np.load(args.pred_file, allow_pickle=True).item()
    predictions = data["predictions"]   # (N, 1, 12, 64, 64)
    ground_truth = data["ground_truth"]  # (N, 1, 12, 64, 64)

    print(f"Predictions shape:  {predictions.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")

    # ── Alpha sweep ──
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    mse_values = np.zeros_like(alphas)

    for i, alpha in enumerate(alphas):
        scaled = alpha * predictions
        mse_values[i] = np.mean((scaled - ground_truth) ** 2)

    # Closed-form optimum
    alpha_cf = np.sum(predictions * ground_truth) / np.sum(predictions ** 2)
    mse_cf = np.mean((alpha_cf * predictions - ground_truth) ** 2)

    # Grid-search optimum
    best_idx = np.argmin(mse_values)
    alpha_grid = alphas[best_idx]
    mse_grid = mse_values[best_idx]

    # MSE at alpha=1 (no scaling)
    mse_baseline = np.mean((predictions - ground_truth) ** 2)

    print(f"\n{'='*50}")
    print(f"Alpha sweep results:")
    print(f"  Baseline (alpha=1.0):  MSE = {mse_baseline:.6f}")
    print(f"  Grid-search best:      alpha = {alpha_grid:.4f}, MSE = {mse_grid:.6f}")
    print(f"  Closed-form optimum:   alpha = {alpha_cf:.4f}, MSE = {mse_cf:.6f}")
    print(f"  MSE improvement:       {(mse_baseline - mse_cf) / mse_baseline * 100:.2f}%")
    print(f"{'='*50}\n")

    # ── Plot MSE vs alpha ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas, mse_values, linewidth=2, label="MSE(alpha)")
    ax.axvline(alpha_cf, color="red", linestyle="--", linewidth=1.5,
               label=f"Optimal alpha = {alpha_cf:.4f}")
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.0,
               label=f"alpha = 1.0 (baseline)")
    ax.scatter([alpha_cf], [mse_cf], color="red", s=80, zorder=5)
    ax.scatter([1.0], [mse_baseline], color="gray", s=80, zorder=5)
    ax.set_xlabel("Alpha", fontsize=13)
    ax.set_ylabel("MSE (mm/hr)^2", fontsize=13)
    ax.set_title("MSE vs Alpha Scaling Factor", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(os.path.dirname(args.pred_file) or ".", "alpha_mse_curve.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved MSE vs alpha plot to {plot_path}")

    # ── Save best-alpha predictions ──
    best_predictions = alpha_cf * predictions
    best_alpha_file = args.pred_file.replace(".npy", "_best_alpha.npy")
    np.save(best_alpha_file, {
        "predictions": best_predictions,
        "ground_truth": ground_truth,
        "alpha": alpha_cf,
        "mse_baseline": mse_baseline,
        "mse_best": mse_cf,
    })
    print(f"Saved best-alpha predictions to {best_alpha_file}")

    # ── Generate GIFs ──
    gif_dir = os.path.join(os.path.dirname(args.pred_file) or ".", "val_alpha_gifs")
    os.makedirs(gif_dir, exist_ok=True)
    num_gifs = min(args.num_gifs, predictions.shape[0])

    cmap = plt.cm.viridis.copy()

    print(f"Generating {num_gifs} prediction vs ground-truth GIFs...")
    for sample_idx in range(num_gifs):
        pred_frames = best_predictions[sample_idx, 0]   # (12, 64, 64)
        gt_frames = ground_truth[sample_idx, 0]          # (12, 64, 64)
        orig_frames = predictions[sample_idx, 0]         # (12, 64, 64) unscaled

        # Per-sample vmax: based on max across all three panels for this sample
        sample_vmax = max(float(orig_frames.max()), float(pred_frames.max()),
                          float(gt_frames.max()), 0.01)
        sample_norm = mcolors.Normalize(vmin=0.0, vmax=sample_vmax)

        frames = []
        for t in range(pred_frames.shape[0]):
            fig, axes = plt.subplots(1, 3, figsize=(16, 4),
                                     gridspec_kw={"right": 0.90})

            # Original prediction (alpha=1)
            axes[0].imshow(orig_frames[t], cmap=cmap, norm=sample_norm, interpolation="nearest")
            axes[0].set_title(f"Pred (alpha=1.0)")
            axes[0].set_xticks([])
            axes[0].set_yticks([])

            # Best alpha prediction
            axes[1].imshow(pred_frames[t], cmap=cmap, norm=sample_norm, interpolation="nearest")
            axes[1].set_title(f"Pred (alpha={alpha_cf:.3f})")
            axes[1].set_xticks([])
            axes[1].set_yticks([])

            # Ground truth
            im = axes[2].imshow(gt_frames[t], cmap=cmap, norm=sample_norm, interpolation="nearest")
            axes[2].set_title("Ground Truth")
            axes[2].set_xticks([])
            axes[2].set_yticks([])

            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
            fig.colorbar(im, cax=cbar_ax, label="Precipitation (mm/hr)")
            fig.suptitle(f"Sample {sample_idx} | t=+{(t+1)*30} min  (vmax={sample_vmax:.2f})", fontsize=13)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            plt.close(fig)
            frames.append(img)

        gif_path = os.path.join(gif_dir, f"sample_{sample_idx}_alpha_compare.gif")
        imageio.mimsave(gif_path, frames, duration=500, loop=0)
        print(f"  Saved {gif_path}")

    print("\nDone.")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Alpha calibration for validation predictions")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ── predict ──
    p_pred = subparsers.add_parser("predict",
        help="Run inference on validation set and save predictions")
    p_pred.add_argument("--config", type=str, required=True,
        help="Path to config YAML file")
    p_pred.add_argument("--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.ckpt)")
    p_pred.add_argument("--num_batches", type=int, default=10,
        help="Number of validation batches to process")
    p_pred.add_argument("--num_samples", type=int, default=5,
        help="Number of ensemble members (averaged for final prediction)")
    p_pred.add_argument("--steps", type=int, default=50,
        help="Number of PLMS denoising steps")
    p_pred.add_argument("--output", type=str, default="val_pred.npy",
        help="Output file path")

    # ── optimize ──
    p_opt = subparsers.add_parser("optimize",
        help="Sweep alpha to minimize MSE, plot curve, save results + GIFs")
    p_opt.add_argument("--pred_file", type=str, default="val_pred.npy",
        help="Path to saved predictions file")
    p_opt.add_argument("--alpha_min", type=float, default=0.1,
        help="Minimum alpha value for sweep")
    p_opt.add_argument("--alpha_max", type=float, default=10.0,
        help="Maximum alpha value for sweep")
    p_opt.add_argument("--alpha_steps", type=int, default=500,
        help="Number of alpha values to sweep")
    p_opt.add_argument("--num_gifs", type=int, default=5,
        help="Number of sample GIFs to generate")

    args = parser.parse_args()

    if args.mode == "predict":
        run_predict(args)
    elif args.mode == "optimize":
        run_optimize(args)


if __name__ == "__main__":
    main()
