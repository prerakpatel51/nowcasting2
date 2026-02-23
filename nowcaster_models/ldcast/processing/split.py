"""Train/validation/test splitting and PyTorch Lightning DataModule.

Copied from ldcast/features/split.py with import path changed to use
local batch.py instead of ldcast.features.batch.

Splitting strategy
------------------
The full timeline is divided into fixed-size chunks (default 2 days).
Chunks are randomly shuffled and assigned to splits (train/valid/test)
in fixed proportions. This "chunk-based" approach avoids leakage between
very nearby frames (which would occur with a random per-frame split) while
still ensuring temporal diversity in each split.

Because the shuffle uses a fixed random seed, the split is deterministic
and reproducible across machines.
"""

from bisect import bisect_left

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from . import batch


def get_chunks(
    primary_raw, valid_frac=0.1, test_frac=0.1,
    chunk_seconds=2*24*60*60, random_seed=None
):
    """Divide the temporal range into fixed-size chunks and assign splits.

    The time axis is partitioned into contiguous windows of `chunk_seconds`
    width. The resulting chunk indices are randomly shuffled (using
    `random_seed` for reproducibility) and then sliced into valid, test,
    and train portions.

    Because chunks are shuffled before slicing, the split is NOT contiguous
    in time — train chunks are interleaved with valid and test chunks across
    the full dataset span. This prevents any systematic temporal bias.

    Parameters
    ----------
    primary_raw : dict
        Patch dictionary for the primary variable (must contain
        "patch_times" and "zero_patch_times" as 1-D int64 arrays of
        Unix timestamps in seconds).
    valid_frac : float
        Fraction of chunks assigned to validation (default 0.10 → 10%).
    test_frac : float
        Fraction of chunks assigned to test (default 0.10 → 10%).
    chunk_seconds : int
        Duration of each temporal chunk in seconds.
        Default 2*24*60*60 = 2 days.
    random_seed : int or None
        Seed for the numpy RNG used to shuffle chunk indices.

    Returns
    -------
    dict
        Keys: "train", "valid", "test".
        Values: sorted list of (t_start, t_end) tuples (Unix seconds),
        one per chunk assigned to that split.
    """
    # Determine the global time range from both data and zero patches
    t0 = min(
        primary_raw["patch_times"][0],
        primary_raw["zero_patch_times"][0]
    )
    t1 = max(
        primary_raw["patch_times"][-1],
        primary_raw["zero_patch_times"][-1]
    )+1

    rng = np.random.RandomState(seed=random_seed)
    # Create chunk boundaries at regular intervals across [t0, t1)
    chunk_limits = np.arange(t0, t1, chunk_seconds)
    num_chunks = len(chunk_limits) - 1

    # Shuffle chunk indices so valid/test chunks are spread across the timeline
    chunk_ind = np.arange(num_chunks)
    rng.shuffle(chunk_ind)

    # First i_valid indices → valid; next i_test → test; rest → train
    i_valid = int(round(num_chunks * valid_frac))
    i_test = i_valid + int(round(num_chunks * test_frac))
    chunk_ind = {
        "valid": chunk_ind[:i_valid],
        "test": chunk_ind[i_valid:i_test],
        "train": chunk_ind[i_test:]
    }

    def get_chunk_limits(chunk_ind_split):
        # Return sorted (t_start, t_end) pairs for the given chunk indices
        return sorted(
            (chunk_limits[i], chunk_limits[i+1])
            for i in chunk_ind_split
        )

    chunks = {
        split: get_chunk_limits(chunk_ind_split)
        for (split, chunk_ind_split) in chunk_ind.items()
    }
    return chunks


def train_valid_test_split(
    raw_data, primary_raw_var, chunks=None, **kwargs
):
    """Split raw patch data into train/valid/test sets by temporal chunks.

    For each split, uses binary search (bisect_left) on the sorted
    patch_times array to efficiently extract the rows whose timestamps
    fall within any of that split's chunks. The same logic is applied
    separately to data patches and zero patches.

    Parameters
    ----------
    raw_data : dict
        Maps variable name → patch dict with keys:
        patches, patch_coords, patch_times, zero_patch_coords,
        zero_patch_times (and optionally other metadata keys).
    primary_raw_var : str
        Variable used to compute chunk boundaries via get_chunks.
    chunks : dict or None
        Pre-computed chunk assignments (output of get_chunks). If None,
        get_chunks is called with **kwargs.
    **kwargs
        Passed to get_chunks: valid_frac, test_frac, chunk_seconds,
        random_seed.

    Returns
    -------
    split_raw_data : dict
        Maps split name → {var → patch dict} for each split.
    chunks : dict
        The chunk assignment used (returned so callers can inspect or reuse).
    """
    if chunks is None:
        primary = raw_data[primary_raw_var]
        chunks = get_chunks(primary, **kwargs)

    def split_chunks_from_array(x, chunks_split, times):
        """Extract rows from array x whose timestamps fall in chunks_split.

        Uses bisect_left for O(log N) chunk boundary lookup, then copies
        each matching row range into a pre-allocated output array.

        Parameters
        ----------
        x : np.ndarray
            Source array; first axis indexes patches.
        chunks_split : list[tuple]
            List of (t_start, t_end) windows for this split.
        times : np.ndarray
            Sorted 1-D array of Unix timestamps aligned with x's first axis.

        Returns
        -------
        np.ndarray
            Same dtype and trailing shape as x, containing only the rows
            whose timestamps fall within chunks_split.
        """
        n = 0
        chunk_ind = []
        for (t0, t1) in chunks_split:
            # Find the slice [k0, k1) whose times fall in [t0, t1)
            k0 = bisect_left(times, t0)
            k1 = bisect_left(times, t1)
            n += k1 - k0
            chunk_ind.append((k0, k1))

        shape = (n,) + x.shape[1:]
        x_chunk = np.empty_like(x, shape=shape)

        j0 = 0
        for (k0, k1) in chunk_ind:
            j1 = j0 + (k1 - k0)
            x_chunk[j0:j1,...] = x[k0:k1,...]
            j0 = j1

        return x_chunk

    split_raw_data = {
        split: {var: {} for var in raw_data}
        for split in chunks
    }
    for (var, raw_data_var) in raw_data.items():
        for (split, chunks_split) in chunks.items():
            # Extract data patch arrays for this split
            split_raw_data[split][var]["patches"] = \
                split_chunks_from_array(
                    raw_data_var["patches"], chunks_split,
                    raw_data_var["patch_times"]
                )
            split_raw_data[split][var]["patch_coords"] = \
                split_chunks_from_array(
                    raw_data_var["patch_coords"], chunks_split,
                    raw_data_var["patch_times"]
                )
            split_raw_data[split][var]["patch_times"] = \
                split_chunks_from_array(
                    raw_data_var["patch_times"], chunks_split,
                    raw_data_var["patch_times"]
                )
            # Extract zero patch arrays for this split
            split_raw_data[split][var]["zero_patch_coords"] = \
                split_chunks_from_array(
                    raw_data_var["zero_patch_coords"], chunks_split,
                    raw_data_var["zero_patch_times"]
                )
            split_raw_data[split][var]["zero_patch_times"] = \
                split_chunks_from_array(
                    raw_data_var["zero_patch_times"], chunks_split,
                    raw_data_var["zero_patch_times"]
                )

            # Pass through any metadata keys not explicitly handled above
            # (e.g. "zero_value", "scale")
            added_keys = set(split_raw_data[split][var].keys())
            missing_keys = set(raw_data_var.keys()) - added_keys
            for k in missing_keys:
                split_raw_data[split][var][k] = raw_data_var[k]

    return (split_raw_data, chunks)


class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the IMERG training pipeline.

    Wraps one BatchGenerator per split (train/valid/test) inside the
    appropriate PyTorch Dataset subclass:
    - Training  → StreamBatchDataset   (random, infinite stream)
    - Valid/Test → DeterministicBatchDataset (fixed seed, reproducible)

    All three dataloaders use batch_size=None because batching is handled
    internally by BatchGenerator (each __getitem__ already returns a full
    batch of shape (B, C, T, H, W)).

    Parameters
    ----------
    variables : dict
        Variable definitions passed to BatchGenerator. Maps variable name
        → dict with "sources", "timesteps", "transform".
    raw : dict
        Per-split patch data. Maps split name → {var → patch dict}.
    predictors : list[str]
        Variable names used as model inputs.
    target : str
        Variable name used as prediction target.
    primary_var : str
        Variable whose raw data drives sampler construction.
    sampling_bins : np.ndarray
        Intensity bin edges for equal-frequency sampling.
    sampler_file : dict
        Maps split name → path for sampler pickle cache.
        E.g. {"train": "cache/sampler_train.pkl", ...}
    batch_size : int
        Samples per batch (default 64).
    train_epoch_size : int
        Batches yielded per training epoch (default 1000).
    valid_epoch_size : int
        Batches in the validation set (default 200).
    test_epoch_size : int
        Batches in the test set (default 1000).
    valid_seed : int or None
        RNG seed for deterministic valid sampling (default None).
    test_seed : int or None
        RNG seed for deterministic test sampling (default None).
    **kwargs
        Additional keyword arguments forwarded to BatchGenerator
        (e.g. interval, sample_shape).
    """

    def __init__(
        self,
        variables, raw, predictors, target, primary_var,
        sampling_bins, sampler_file,
        batch_size=64,
        train_epoch_size=1000, valid_epoch_size=200, test_epoch_size=1000,
        valid_seed=None, test_seed=None,
        **kwargs
    ):
        super().__init__()
        # Create one BatchGenerator per split; only the train generator uses augmentation
        self.batch_gen = {
            split: batch.BatchGenerator(
                variables, raw_var, predictors, target, primary_var,
                sampling_bins=sampling_bins, batch_size=batch_size,
                sampler_file=sampler_file.get(split),
                augment=(split=="train"),
                **kwargs
            )
            for (split, raw_var) in raw.items()
        }

        # Wrap each generator in the appropriate Dataset subclass
        self.datasets = {}
        if "train" in self.batch_gen:
            # StreamBatchDataset: yields new random batches each epoch
            self.datasets["train"] = batch.StreamBatchDataset(
                self.batch_gen["train"], train_epoch_size
            )
        if "valid" in self.batch_gen:
            # DeterministicBatchDataset: pre-sampled with fixed seed
            self.datasets["valid"] = batch.DeterministicBatchDataset(
                self.batch_gen["valid"], valid_epoch_size, random_seed=valid_seed
            )
        if "test" in self.batch_gen:
            # DeterministicBatchDataset: pre-sampled with fixed seed
            self.datasets["test"] = batch.DeterministicBatchDataset(
                self.batch_gen["test"], test_epoch_size, random_seed=test_seed
            )

    def dataloader(self, split):
        """Return a DataLoader for the requested split.

        batch_size=None because BatchGenerator already yields full batches.
        pin_memory=True to speed up CPU→GPU transfer during training.
        num_workers=0 because Numba JIT does not survive subprocess forking.
        """
        return DataLoader(
            self.datasets[split], batch_size=None,
            pin_memory=True, num_workers=0
        )

    def train_dataloader(self):
        """Return the training DataLoader (random, one StreamBatchDataset)."""
        return self.dataloader("train")

    def val_dataloader(self):
        """Return the validation DataLoader (deterministic, fixed seed 1234)."""
        return self.dataloader("valid")

    def test_dataloader(self):
        """Return the test DataLoader (deterministic, fixed seed 2345)."""
        return self.dataloader("test")
