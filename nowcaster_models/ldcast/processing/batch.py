"""Batch generation and PyTorch dataset wrappers for training.

Copied from ldcast/features/batch.py with import paths changed to use
local patches.py and sampling.py instead of ldcast.features.

Key classes
-----------
BatchGenerator
    Owns the patch index and sampler for one data split. Assembles batches
    of (predictor, target) tensors on demand.
PatchIndex
    Hash-map from (time, patch_row, patch_col) → array index. Used by
    build_batch to assemble a spatial tile of patches for each timestep.
StreamBatchDataset
    Wraps BatchGenerator as a PyTorch IterableDataset for the training
    split (random, infinite stream).
DeterministicBatchDataset
    Wraps BatchGenerator as a PyTorch Dataset for validation/test splits;
    all samples are pre-drawn with a fixed random seed so results are
    reproducible across runs.
"""

from datetime import datetime, timedelta
import os
import pickle

from numba import njit, prange, types
from numba.typed import Dict
import numpy as np
from torch.utils.data import Dataset, IterableDataset

from ..io.patches import unpack_patches
from .sampling import EqualFrequencySampler


class BatchGenerator:
    """Generates training batches from pre-extracted patch data.

    Owns:
    - A PatchIndex per raw variable (fast (t, i, j) → pixel lookup).
    - An EqualFrequencySampler that proposes (t0, i0, j0) anchor points
      with equal frequency across precipitation intensity bins.

    Parameters
    ----------
    variables : dict
        Maps variable name → dict with keys:
            "sources"   : list of raw variable names to pull from.
            "timesteps" : np.ndarray of integer timestep offsets.
            "transform" : callable that normalizes raw pixel arrays.
    raw : dict
        Maps raw variable name → patch data dict (patches, patch_coords,
        patch_times, zero_patch_coords, zero_patch_times).
    predictors : list[str]
        Variable names used as model inputs (e.g. ["IMERG-O"]).
    target : str
        Variable name used as the prediction target (e.g. "IMERG-T").
    primary_var : str
        Variable whose raw data is used to build the sampler.
    time_range_sampling : tuple[int, int]
        Relative timestep window (in multiples of interval) that the sampler
        searches for valid anchor positions. Default (-1, 2).
    sampling_bins : np.ndarray or None
        Intensity bin edges for equal-frequency sampling.
    sampler_file : str or None
        Path to a pickle cache of the EqualFrequencySampler. Built and saved
        on first run; loaded on subsequent runs to avoid recomputation.
    sample_shape : tuple[int, int]
        Spatial box size in patch units, e.g. (2, 2) → 2×2 patches.
    batch_size : int
        Number of samples per batch.
    interval : timedelta
        Temporal spacing between frames (30 min for IMERG).
    random_seed : int or None
        Seed for the batch-level RNG (used for augmentation).
    augment : bool
        Whether to apply random spatial augmentation (train split only).
    """

    def __init__(self,
        variables,
        raw,
        predictors,
        target,
        primary_var,
        time_range_sampling=(-1,2),
        forecast_raw_vars=(),
        sampling_bins=None,
        sampler_file=None,
        sample_shape=(4,4),
        batch_size=32,
        interval=timedelta(minutes=5),
        random_seed=None,
        augment=False,
        dry_bin_weight=0.0
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dry_bin_weight = dry_bin_weight
        self.interval = interval
        self.interval_secs = np.int64(self.interval.total_seconds())
        self.variables = variables
        self.predictors = predictors
        self.target = target
        # Union of all raw source names needed by predictors + target
        self.used_variables = predictors + [target]
        self.rng = np.random.RandomState(seed=random_seed)
        self.augment = augment

        # Collect unique raw source names across all used variables
        self.sources = set.union(
            *(set(variables[v]["sources"]) for v in self.used_variables)
        )
        self.forecast_raw_vars = set(forecast_raw_vars) & self.sources

        # Build a PatchIndex for every raw source variable
        self.patch_index = {}
        for raw_name_base in self.sources:
            if raw_name_base in forecast_raw_vars:
                # NWP forecast variables come in multiple lag variants (e.g. "IFS-0", "IFS-6")
                raw_names = (
                    rn for rn in raw if rn.startswith(raw_name_base+"-")
                )
            else:
                raw_names = (raw_name_base,)
            for raw_name in raw_names:
                raw_data = raw[raw_name]
                self.setup_index(raw_name, raw_data, sample_shape)

        # Wrap multi-lag NWP variables in a ForecastPatchIndexWrapper
        for raw_name in self.forecast_raw_vars:
            patch_index_var = {
                k: v for (k,v) in self.patch_index.items()
                if k.startswith(raw_name+"-")
            }
            self.patch_index[raw_name] = \
                ForecastPatchIndexWrapper(patch_index_var)

        # Build or load the equal-frequency sampler
        if (sampler_file is None) or not os.path.isfile(sampler_file):
            print("No cached sampler found, creating a new one...")
            primary_raw_var = variables[primary_var]["sources"][0]

            # Determine the valid time range: covers all timesteps of all variables
            t0 = t1 = None
            for (var_name, var_data) in variables.items():
                # Take the first and last timestep, offset by -1 on the lower end
                timesteps = var_data["timesteps"][[0,-1]].copy()
                timesteps[0] -= 1
                ts_secs = timesteps * \
                    var_data.get("timestep_secs", self.interval_secs)
                timesteps = ts_secs // self.interval_secs
                t0 = timesteps[0] if t0 is None else min(t0,timesteps[0])
                t1 = timesteps[-1] if t1 is None else max(t1,timesteps[-1])
            time_range_valid = (t0,t1+1)

            self.sampler = EqualFrequencySampler(
                sampling_bins, raw[primary_raw_var],
                self.patch_index[primary_raw_var], sample_shape,
                time_range_valid, time_range_sampling=time_range_sampling,
                timestep_secs=self.interval_secs
            )
            if sampler_file is not None:
                print(f"Caching sampler to {sampler_file}.")
                os.makedirs(os.path.dirname(sampler_file), exist_ok=True)
                with open(sampler_file, 'wb') as f:
                    pickle.dump(self.sampler, f)
        else:
            print(f"Loading cached sampler from {sampler_file}.")
            with open(sampler_file, 'rb') as f:
                self.sampler = pickle.load(f)

        # Set dry bin weight on the sampler (works with both fresh and cached samplers)
        self.sampler.dry_bin_weight = self.dry_bin_weight

    def setup_index(self, raw_name, raw_data, box_size):
        """Create a PatchIndex for a single raw data variable.

        Parameters
        ----------
        raw_name : str
            Key used to store the index in self.patch_index.
        raw_data : dict
            Patch dictionary (patches, patch_coords, patch_times, ...).
        box_size : tuple[int, int]
            Spatial box in patch units (rows, cols); e.g. (2, 2).
        """
        zero_value = raw_data.get("zero_value", 0)
        missing_value = raw_data.get("missing_value", zero_value)

        self.patch_index[raw_name] = PatchIndex(
            *unpack_patches(raw_data),
            zero_value=zero_value,
            missing_value=missing_value,
            interval=self.interval,
            box_size=box_size
        )

    def augmentations(self):
        """Sample a random set of augmentation flags.

        Returns
        -------
        tuple[int, int, int]
            (transpose, flipud, fliplr), each 0 or 1, drawn once per batch.
        """
        return tuple(self.rng.randint(2, size=3))

    def augment_batch(self, batch, transpose, flipud, fliplr):
        """Apply spatial augmentation to a batch array.

        All three operations act only on the last two axes (H, W) so the
        same transform is applied consistently across the time dimension.

        Parameters
        ----------
        batch : np.ndarray
            Shape (..., H, W).
        transpose : int
            1 → swap H and W axes.
        flipud : int
            1 → flip along H axis.
        fliplr : int
            1 → flip along W axis.

        Returns
        -------
        np.ndarray
            Augmented copy of the batch.
        """
        if self.augment:
            if transpose:
                axes = list(range(batch.ndim))
                axes = axes[:-2] + [axes[-1], axes[-2]]
                batch = batch.transpose(axes)
            flips = []
            if flipud:
                flips.append(-2)
            if fliplr:
                flips.append(-1)
            if flips:
                batch = np.flip(batch, axis=flips)
        return batch.copy()

    def batch(self, samples=None, batch_size=None):
        """Assemble one batch of (predictors, target) arrays.

        Flow
        ----
        1. Call sampler → (t0, i0, j0) anchor for each sample.
        2. Expand t0 into absolute timestamps for each variable's timesteps.
        3. Call PatchIndex to assemble pixel arrays for each variable.
        4. Apply the variable's transform (threshold → log10 → normalize).
        5. Optionally augment (training only).

        Parameters
        ----------
        samples : np.ndarray or None
            Pre-drawn (t0, i0, j0) array of shape (B, 3). If None, the
            sampler is called to draw B new samples.
        batch_size : int or None
            Number of samples. Defaults to self.batch_size.

        Returns
        -------
        pred_batch : list[tuple[np.ndarray, np.ndarray]]
            One entry per predictor variable: (data, t_relative).
            data shape: (B, 1, T_in, H, W), t_relative shape: (B, T_in).
        target_batch : np.ndarray
            Shape (B, 1, T_out, H, W) — normalized target frames.
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Draw anchor positions if not provided externally (e.g. DeterministicBatchDataset)
        if samples is None:
            samples = self.sampler(batch_size)
        (t0,i0,j0) = samples.T

        # Draw augmentation flags once for the whole batch (same transform across variables)
        if self.augment:
            augmentations = self.augmentations()

        batch = {}
        for var_name in self.used_variables:
            var_data = self.variables[var_name]

            # Compute absolute timestamps for every timestep of this variable.
            # t_shift aligns t0 to the variable's own temporal grid (handles
            # variables with a coarser interval than self.interval).
            ts_secs = var_data.get("timestep_secs", self.interval_secs)
            t_shift = -(t0 % ts_secs)
            t0_shifted = t0 + t_shift
            # Shape: (B, T) — one timestamp per sample per timestep
            t = t0_shifted[:,None] + ts_secs*var_data["timesteps"][None,:]
            # Relative offsets in units of interval_secs (used by the model's time embedding)
            t_relative = (t - t0[:,None]) / self.interval_secs

            # Look up pixel data for each source variable via PatchIndex
            raw_data = (
                self.patch_index[raw_name](t,i0,j0)
                for raw_name in var_data["sources"]
            )

            # Apply threshold → log10 → normalize transform
            batch_var = var_data["transform"](*raw_data)

            # Insert a channel dimension if not already present (channels-first format)
            add_dims = (1,) if batch_var.ndim == 4 else ()
            batch_var = np.expand_dims(batch_var, add_dims)

            if self.augment:
                batch_var = self.augment_batch(batch_var, *augmentations)

            batch[var_name] = (batch_var, t_relative.astype(np.float32))

        pred_batch = [batch[v] for v in self.predictors]
        target_batch = batch[self.target][0]
        return (pred_batch, target_batch)

    def batches(self, *args, num=None, **kwargs):
        """Yield batches indefinitely (or up to `num` times).

        Parameters
        ----------
        num : int or None
            If given, stop after this many batches. Otherwise yield forever.
        """
        if num is not None:
            for i in range(num):
                yield self.batch(*args, **kwargs)
        else:
            while True:
                yield self.batch(*args, **kwargs)


class StreamBatchDataset(IterableDataset):
    """PyTorch IterableDataset for the training split.

    Each call to __iter__ yields `batches_per_epoch` randomly sampled
    batches from the BatchGenerator. Because it is an IterableDataset,
    the DataLoader must be used with batch_size=None (batching is handled
    internally by BatchGenerator).

    Parameters
    ----------
    batch_gen : BatchGenerator
        The generator to draw batches from.
    batches_per_epoch : int
        How many batches constitute one "epoch". The sampler continues
        across epoch boundaries — there is no epoch-level reset.
    """

    def __init__(self, batch_gen, batches_per_epoch):
        super().__init__()
        self.batch_gen = batch_gen
        self.batches_per_epoch = batches_per_epoch

    def __iter__(self):
        batches = self.batch_gen.batches(num=self.batches_per_epoch)
        yield from batches


class DeterministicBatchDataset(Dataset):
    """PyTorch Dataset for validation and test splits.

    All (t0, i0, j0) anchor positions are sampled once at construction
    time using a fixed random seed, so the same batch is returned for
    __getitem__(i) every time. This guarantees reproducible validation
    and test metrics across runs.

    Parameters
    ----------
    batch_gen : BatchGenerator
        The generator used to assemble pixel arrays given pre-drawn samples.
    batches_per_epoch : int
        Total number of batches in the split.
    random_seed : int or None
        Seed used to reset the sampler's RNG before pre-drawing samples.
    """

    def __init__(self, batch_gen, batches_per_epoch, random_seed=None):
        super().__init__()
        self.batch_gen = batch_gen
        self.batches_per_epoch = batches_per_epoch
        # Fix the sampler's RNG so all positions are drawn deterministically
        self.batch_gen.sampler.rng = np.random.RandomState(seed=random_seed)
        # Pre-draw all anchor positions upfront
        self.samples = [
            self.batch_gen.sampler(self.batch_gen.batch_size)
            for i in range(self.batches_per_epoch)
        ]

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, ind):
        """Return the pre-drawn batch at index `ind`."""
        return self.batch_gen.batch(samples=self.samples[ind])


class PatchIndex:
    """Fast lookup structure mapping (time, patch_row, patch_col) → pixel data.

    At construction, builds a Numba typed Dict from (t, i, j) → array index.
    Data patches map to their row in self.patch_data; zero patches map to
    IDX_ZERO (-1); positions absent from the index map to IDX_MISSING (-2).

    When called, assembles a full (batch_size, num_timesteps, H, W) array
    by stitching together the 2×2 (or sample_shape) grid of 32×32 patches
    for each (sample, timestep) pair. The stitching is done by the Numba
    JIT kernel build_batch which runs one parallel thread per sample.

    Parameters
    ----------
    patch_data : np.ndarray
        Shape (N, patch_h, patch_w) — pixel arrays for non-zero patches.
    patch_coords : np.ndarray
        Shape (N, 2) — (row, col) grid coordinates for each data patch.
    patch_times : np.ndarray
        Shape (N,) — Unix timestamp (seconds) for each data patch.
    zero_patch_coords : np.ndarray
        Shape (M, 2) — coordinates of dry/zero patches.
    zero_patch_times : np.ndarray
        Shape (M,) — timestamps of dry/zero patches.
    interval : timedelta
        Frame interval (30 min for IMERG).
    box_size : tuple[int, int]
        Spatial box in patch units (rows, cols), e.g. (2, 2).
    zero_value : scalar
        Fill value for zero patches (typically 0.0).
    missing_value : scalar
        Fill value for positions absent from the index (typically 0.0).
    """

    # Sentinel index values stored in the lookup dict
    IDX_ZERO = -1    # patch exists but is all-zero (dry)
    IDX_MISSING = -2 # patch not present in the dataset at all

    def __init__(
        self, patch_data, patch_coords, patch_times,
        zero_patch_coords, zero_patch_times,
        interval=timedelta(minutes=5),
        box_size=(4,4), zero_value=0,
        missing_value=0
    ):
        self.dt = int(round(interval.total_seconds()))
        self.box_size = box_size
        self.zero_value = zero_value
        self.missing_value = missing_value
        self.patch_data = patch_data
        # Output pixel dimensions: each patch tile is (patch_h × patch_w),
        # and we assemble a (box_rows × box_cols) grid of them.
        self.sample_shape = (
            self.patch_data.shape[1]*box_size[0],
            self.patch_data.shape[2]*box_size[1]
        )

        # Build the Numba typed Dict: (t, i, j) → array index
        self.patch_index = Dict.empty(
            key_type=types.UniTuple(types.int64, 3),
            value_type=types.int64
        )
        init_patch_index(self.patch_index, patch_coords, patch_times)
        # Zero patches use sentinel IDX_ZERO so build_batch fills them with zero_value
        init_patch_index_zero(self.patch_index, zero_patch_coords,
            zero_patch_times, PatchIndex.IDX_ZERO)

        # Pre-allocated output buffer; resized only when batch shape grows
        self._batch = None

    def _alloc_batch(self, batch_size, num_timesteps):
        """Allocate (or reuse) the output buffer for build_batch.

        The buffer is reused across calls to avoid repeated allocation.
        It is only reallocated when the requested shape is larger than
        the current buffer.
        """
        needs_rebuild = (self._batch is None) or \
            (self._batch.shape[0] < batch_size) or \
            (self._batch.shape[1] < num_timesteps)
        if needs_rebuild:
            del self._batch
            self._batch = np.zeros(
                (batch_size,num_timesteps)+self.sample_shape,
                self.patch_data.dtype
            )
        return self._batch

    def __call__(self, t, i0_all, j0_all):
        """Assemble pixel arrays for a batch of (time, position) requests.

        Parameters
        ----------
        t : np.ndarray
            Shape (B, T) — absolute Unix timestamps for each (sample, timestep).
        i0_all : np.ndarray
            Shape (B,) — top-left patch row for each sample.
        j0_all : np.ndarray
            Shape (B,) — top-left patch col for each sample.

        Returns
        -------
        np.ndarray
            Shape (B, T, H, W) where H = patch_h × box_rows,
            W = patch_w × box_cols. Each spatial tile is stitched from
            box_rows × box_cols patches looked up in the index.
        """
        batch = self._alloc_batch(*t.shape)

        # Compute the exclusive end of the spatial box
        i1_all = i0_all + self.box_size[0]
        j1_all = j0_all + self.box_size[1]
        bi_size = self.patch_data.shape[1]  # pixel height of one patch
        bj_size = self.patch_data.shape[2]  # pixel width  of one patch

        # Numba parallel kernel: one thread per sample in the batch
        build_batch(batch, self.patch_data, self.patch_index,
            t, i0_all, i1_all, j0_all, j1_all,
            bi_size, bj_size, self.zero_value,
            self.missing_value)

        return batch[:,:t.shape[1],...]


@njit
def init_patch_index(patch_index, patch_coords, patch_times):
    """Populate the lookup dict for data (non-zero) patches.

    Maps (t, i, j) → k, where k is the row index into patch_data.

    Parameters
    ----------
    patch_index : numba.typed.Dict
        Target dict keyed by (int64, int64, int64).
    patch_coords : np.ndarray
        Shape (N, 2) — (row, col) for each patch.
    patch_times : np.ndarray
        Shape (N,) — Unix timestamp for each patch.
    """
    for k in range(patch_coords.shape[0]):
        t = patch_times[k]
        i = np.int64(patch_coords[k,0])
        j = np.int64(patch_coords[k,1])
        patch_index[(t,i,j)] = k


@njit
def init_patch_index_zero(patch_index, zero_patch_coords,
    zero_patch_times, idx_zero):
    """Populate the lookup dict for zero (dry) patches.

    Maps (t, i, j) → idx_zero (sentinel -1). build_batch uses this
    sentinel to fill the output with zero_value instead of reading
    from patch_data.

    Parameters
    ----------
    patch_index : numba.typed.Dict
        Target dict (shared with init_patch_index).
    zero_patch_coords : np.ndarray
        Shape (M, 2) — coordinates of dry patches.
    zero_patch_times : np.ndarray
        Shape (M,) — timestamps of dry patches.
    idx_zero : int
        Sentinel value (PatchIndex.IDX_ZERO = -1).
    """
    for k in range(zero_patch_coords.shape[0]):
        t = zero_patch_times[k]
        i = np.int64(zero_patch_coords[k,0])
        j = np.int64(zero_patch_coords[k,1])
        patch_index[(t,i,j)] = idx_zero


IDX_ZERO = PatchIndex.IDX_ZERO
IDX_MISSING = PatchIndex.IDX_MISSING

@njit(parallel=True)
def build_batch(
    batch, patch_data, patch_index,
    t_all, i0_all, i1_all, j0_all, j1_all,
    bi_size, bj_size, zero_value, missing_value
):
    """Assemble a batch of spatially tiled, temporally stacked pixel arrays.

    For each sample k (parallelised with prange), iterates over every
    requested timestep and every patch in the spatial box, placing the
    correct 32×32 tile into the pre-allocated output buffer.

    Index outcomes
    --------------
    ind >= 0   : real data patch → copy from patch_data[ind]
    IDX_ZERO   : dry patch       → fill with zero_value
    IDX_MISSING: absent patch    → fill with missing_value

    Parameters
    ----------
    batch : np.ndarray
        Pre-allocated output, shape (B, T_max, H, W). Modified in place.
    patch_data : np.ndarray
        Shape (N, patch_h, patch_w) — all data patch pixels.
    patch_index : numba.typed.Dict
        Maps (t, i, j) → array index (or sentinel).
    t_all : np.ndarray
        Shape (B, T) — absolute timestamps per (sample, timestep).
    i0_all, i1_all : np.ndarray
        Shape (B,) — row range [i0, i1) of the spatial box per sample.
    j0_all, j1_all : np.ndarray
        Shape (B,) — col range [j0, j1) of the spatial box per sample.
    bi_size, bj_size : int
        Pixel dimensions of a single patch (32, 32 for IMERG).
    zero_value, missing_value : scalar
        Fill values for the two sentinel cases.
    """
    for k in prange(t_all.shape[0]):        # parallel over batch samples
        i0 = i0_all[k]
        i1 = i1_all[k]
        j0 = j0_all[k]
        j1 = j1_all[k]

        for (bt,t) in enumerate(t_all[k,:]):   # iterate over timesteps
            for i in range(i0, i1):             # iterate over patch rows in box
                # Compute pixel row offset for this patch within the output tile
                bi0 = (i-i0) * bi_size
                bi1 = bi0 + bi_size
                for j in range(j0, j1):         # iterate over patch cols in box
                    ind = int(patch_index.get((t,i,j), IDX_MISSING))
                    # Compute pixel col offset for this patch within the output tile
                    bj0 = (j-j0) * bj_size
                    bj1 = bj0 + bj_size
                    if ind >= 0:
                        # Real data patch: copy pixel data from the store
                        batch[k,bt,bi0:bi1,bj0:bj1] = patch_data[ind]
                    elif ind == IDX_ZERO:
                        # Dry patch: fill with the zero sentinel value
                        batch[k,bt,bi0:bi1,bj0:bj1] = zero_value
                    elif ind == IDX_MISSING:
                        # Missing patch: fill with the missing sentinel value
                        batch[k,bt,bi0:bi1,bj0:bj1] = missing_value


class ForecastPatchIndexWrapper(PatchIndex):
    """Wrapper that selects the correct NWP forecast lag at each timestep.

    NWP forecasts (e.g. IFS) are stored under names like "IFS-0", "IFS-6",
    "IFS-12", ... indicating the forecast lead time in hours at which the
    data was issued. For a given anchor time t0, this wrapper selects the
    appropriate lag variant for each timestep so that data is always taken
    from the nearest-prior forecast run.

    Parameters
    ----------
    patch_index : dict
        Maps lag-suffixed names (e.g. "IFS-0", "IFS-6") to PatchIndex objects.
    """

    def __init__(self, patch_index):
        self.patch_index = patch_index
        raw_names = {"-".join(v.split("-")[:-1]) for v in patch_index}
        if len(raw_names) != 1:
            raise ValueError(
                "Can only wrap variables with the same base name")
        self.raw_name = list(raw_names)[0]
        lags_hour = [int(v.split("-")[-1]) for v in patch_index]
        self.lags_hour = set(lags_hour)
        forecast_interval_hour = np.diff(sorted(lags_hour))
        if len(set(forecast_interval_hour)) != 1:
            raise ValueError("Lags must be evenly spaced")
        forecast_interval_hour = forecast_interval_hour[0]
        if (24 % forecast_interval_hour):
            raise ValueError(
                "24 hours must be a multiple of the forecast interval")
        self.forecast_interval_hour = forecast_interval_hour
        self.forecast_interval = 3600 * forecast_interval_hour

        self._batch = None
        v = list(self.patch_index.keys())[0]
        self.sample_shape = self.patch_index[v].sample_shape
        self.patch_data = self.patch_index[v].patch_data

    def __call__(self, t, i0, j0):
        """Assemble pixel arrays, picking the right forecast lag per timestep."""
        batch = self._alloc_batch(*t.shape)

        # Determine which forecast lag applies at each (sample, timestep)
        t0 = t[:,:1]
        start_time_from_fc = t0 % self.forecast_interval
        time_from_fc = start_time_from_fc + (t - t0)
        lags_hour = (time_from_fc // self.forecast_interval) * \
            self.forecast_interval_hour

        # Fill the output buffer lag by lag, masking non-matching timesteps
        for lag in self.lags_hour:
            raw_name_lag = f"{self.raw_name}-{lag}"
            batch_lag = self.patch_index[raw_name_lag](t,i0,j0)
            lag_mask = (lags_hour == lag)
            copy_masked_times(batch_lag, batch, lag_mask)

        return batch[:,:t.shape[1],...]


@njit(parallel=True)
def copy_masked_times(from_batch, to_batch, mask):
    """Copy timestep frames from from_batch to to_batch where mask is True.

    Parameters
    ----------
    from_batch : np.ndarray
        Shape (B, T, H, W) — source frames.
    to_batch : np.ndarray
        Shape (B, T, H, W) — destination buffer (modified in place).
    mask : np.ndarray
        Shape (B, T) — boolean mask; True where the copy should happen.
    """
    for k in prange(from_batch.shape[0]):
        for bt in range(from_batch.shape[1]):
            if mask[k,bt]:
                to_batch[k,bt,:,:] = from_batch[k,bt,:,:]
