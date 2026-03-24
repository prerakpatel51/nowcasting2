"""Compute normalization statistics from the FULL IMERG dataset.

Computes mean and std of log10(precipitation) across ALL pixels,
including zero/dry patches (clamped to fill_value before log10).
This gives a normalization that properly centers the full data
distribution, rather than only the precipitating pixels.

Usage (from repo root):
    python -m nowcaster_models.ldcast.compute_normalization --config nowcaster_models/ldcast/config_improved.yaml
"""

import argparse
import os

import numpy as np
from omegaconf import OmegaConf

from .io import patches


def compute_stats(config):
    """Compute mean and std of the full transformed distribution."""
    var = config.data.var_name
    threshold = config.transform.threshold
    fill_value = config.transform.fill_value

    print(f"Loading patches for {var} from {config.data.patch_dir}...")
    raw = patches.load_all_patches(config.data.patch_dir, var)

    data_patches = raw["patches"]          # (N_data, 32, 32)
    n_zero = raw["zero_patch_coords"].shape[0]
    patch_pixels = data_patches.shape[1] * data_patches.shape[2]  # 1024

    print(f"  Data patches:  {data_patches.shape[0]}")
    print(f"  Zero patches:  {n_zero}")
    print(f"  Pixels/patch:  {patch_pixels}")

    # --- Process data patches in chunks to limit memory ---
    chunk_size = 10000
    n_patches = data_patches.shape[0]
    sum_x = 0.0
    sum_x2 = 0.0
    n_data_pixels = 0

    print("Computing statistics over data patches...")
    for start in range(0, n_patches, chunk_size):
        end = min(start + chunk_size, n_patches)
        chunk = data_patches[start:end].astype(np.float64).ravel()

        # Step 1: clamp sub-threshold to fill_value
        chunk[chunk < threshold] = fill_value

        # Step 2: log10
        chunk = np.log10(chunk)

        sum_x += chunk.sum()
        sum_x2 += (chunk ** 2).sum()
        n_data_pixels += chunk.shape[0]

        if (start // chunk_size) % 20 == 0:
            print(f"  Processed {end}/{n_patches} data patches...")

    # --- Add zero patches (all pixels = 0 -> fill_value -> log10) ---
    zero_log = np.log10(fill_value)  # log10(0.01) = -2.0
    n_zero_pixels = n_zero * patch_pixels
    sum_x += n_zero_pixels * zero_log
    sum_x2 += n_zero_pixels * (zero_log ** 2)

    n_total = n_data_pixels + n_zero_pixels

    mean = sum_x / n_total
    var = sum_x2 / n_total - mean ** 2
    std = np.sqrt(var)

    return mean, std, n_total


def main():
    parser = argparse.ArgumentParser(
        description="Compute normalization mean/std from full IMERG distribution")
    parser.add_argument("--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config_improved.yaml"),
        help="Path to config YAML file")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    mean, std, n_total = compute_stats(config)

    print(f"\n{'='*50}")
    print(f"Full-distribution normalization statistics")
    print(f"{'='*50}")
    print(f"  Total pixels:  {n_total:,}")
    print(f"  mean:          {mean:.6f}")
    print(f"  std:           {std:.6f}")
    print(f"\nUpdate your config YAML with:")
    print(f"  transform:")
    print(f"    mean: {mean:.6f}")
    print(f"    std: {std:.6f}")
    print(f"\nFor reference, after this normalization:")
    print(f"  Dry pixels (0 mm/hr) will be at: {(np.log10(0.01) - mean) / std:.2f} sigma")
    print(f"  Threshold (0.1 mm/hr) will be at: {(np.log10(0.1) - mean) / std:.2f} sigma")
    print(f"  Moderate (1.0 mm/hr) will be at: {(np.log10(1.0) - mean) / std:.2f} sigma")


if __name__ == "__main__":
    main()
