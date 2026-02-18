"""IMERG training data setup using the ldcast pipeline.

Mirrors ldcast/scripts/train_nowcaster.py:setup_data() with IMERG parameters.
All configuration is read from config.yaml via OmegaConf.
"""

import os
from datetime import timedelta

import numpy as np
from omegaconf import OmegaConf

from . import patches
from . import split
from . import transform


def setup_data(config):
    """Set up the IMERG data pipeline and return a DataModule.

    Args:
        config: OmegaConf config object loaded from config.yaml.

    Returns:
        split.DataModule: PyTorch Lightning DataModule with train/valid/test
        dataloaders ready for training.
    """
    var = config.data.var_name  # "IMERG"

    # Load patches
    print(f"Loading patches for {var} from {config.data.patch_dir}...")
    raw = {var: patches.load_all_patches(config.data.patch_dir, var)}
    print(f"  Data patches: {raw[var]['patches'].shape[0]}")
    print(f"  Zero patches: {raw[var]['zero_patch_coords'].shape[0]}")

    # Split into train/valid/test
    print("Splitting into train/valid/test...")
    (raw, chunks) = split.train_valid_test_split(
        raw, var,
        chunk_seconds=config.split.chunk_days * 86400,
        valid_frac=config.split.valid_frac,
        test_frac=config.split.test_frac,
        random_seed=config.split.random_seed
    )
    for s in ["train", "valid", "test"]:
        n_data = raw[s][var]["patches"].shape[0]
        n_zero = raw[s][var]["zero_patch_coords"].shape[0]
        print(f"  {s}: {n_data} data patches, {n_zero} zero patches")

    # Transform: threshold + log10 + normalize (same as ldcast's normalize_threshold)
    transform_fn = transform.normalize_threshold(
        log=config.transform.log,
        threshold=config.transform.threshold,
        fill_value=config.transform.fill_value,
        mean=config.transform.mean,
        std=config.transform.std
    )

    # Variable definitions
    # Target: future timesteps 1..12 (30-min intervals = 0.5h to 6h ahead)
    # Observation: past timesteps -15..0 (16 input frames)
    variables = {
        f"{var}-T": {
            "sources": [var],
            "timesteps": np.arange(1, config.pipeline.output_timesteps + 1),
            "transform": transform_fn,
        },
        f"{var}-O": {
            "sources": [var],
            "timesteps": np.arange(-config.pipeline.input_timesteps + 1, 1),
            "transform": transform_fn,
        }
    }

    # Sampling bins: log-spaced intensity bins for equal-frequency sampling
    bins = np.exp(np.linspace(
        np.log(config.sampling.bins_low),
        np.log(config.sampling.bins_high),
        config.sampling.num_bins
    ))

    # Sampler cache files
    cache_dir = config.data.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    sampler_file = {
        "train": os.path.join(cache_dir, "sampler_train.pkl"),
        "valid": os.path.join(cache_dir, "sampler_valid.pkl"),
        "test": os.path.join(cache_dir, "sampler_test.pkl"),
    }

    # Create DataModule
    print("Creating DataModule...")
    datamodule = split.DataModule(
        variables, raw,
        predictors=[f"{var}-O"],
        target=f"{var}-T",
        primary_var=f"{var}-T",
        sampling_bins=bins,
        batch_size=config.sampling.batch_size,
        interval=timedelta(minutes=config.pipeline.interval_minutes),
        sample_shape=tuple(config.pipeline.sample_shape),
        sampler_file=sampler_file,
        valid_seed=1234,
        test_seed=2345
    )
    print("DataModule ready.")
    return datamodule
