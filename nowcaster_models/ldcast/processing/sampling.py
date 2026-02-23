"""Stratified (equal-frequency) sampling for IMERG training data.

Copied from ldcast/features/sampling.py with import path changed to use
local patches.py instead of ldcast.features.patches.

The core problem this module solves
-------------------------------------
IMERG data is heavily imbalanced: the majority of patches are dry or
near-zero precipitation. If samples were drawn uniformly at random, the
model would almost never see heavy rain during training. The
EqualFrequencySampler addresses this by:

1. Classifying every patch into one of N intensity bins based on the
   99th-percentile pixel value (bin_classify_patches_parallel).
2. For each bin, finding all valid "starting positions" — anchor points
   (t, i, j) from which a complete spatial+temporal window can be drawn
   without any missing patches (indices_with_complete_sample,
   starting_indices_for_centers).
3. At sample time, picking a bin uniformly at random and then sampling
   uniformly within that bin → equal representation across intensities.
"""

from bisect import bisect_left
import multiprocessing
import dask
from numba import njit, prange, types
from numba.typed import Dict
import numpy as np
from ..io.patches import unpack_patches


class EqualFrequencySampler:
    """Samples training patches with equal probability per intensity bin.

    Constructed once per data split, then called repeatedly at training
    time to produce batches of (t0, i0, j0) anchor coordinates.

    Construction is expensive (minutes for large datasets) because it
    requires classifying every patch and checking spatial/temporal
    completeness. Results are cached in a pickle file by BatchGenerator.

    Parameters
    ----------
    bins : np.ndarray
        Monotonically increasing bin edges, shape (num_bins,). Values
        between bins[k] and bins[k+1] land in bin k.
        Example: np.exp(np.linspace(np.log(0.1), np.log(20), 10))
    patch_data : dict
        Full patch dictionary for the primary variable (e.g. IMERG).
        Must contain: patches, patch_coords, patch_times,
        zero_patch_coords, zero_patch_times.
    patch_index : PatchIndex
        Pre-built index for the primary variable (used for completeness
        checking).
    sample_shape : tuple[int, int]
        Spatial box size in patch units, e.g. (2, 2).
    time_range_valid : tuple[int, int]
        Closed-open range of relative timestep offsets (in multiples of
        interval) that must be present for a position to be valid.
        E.g. (-16, 13) covers 16 input + 12 output frames.
    time_range_sampling : tuple[int, int] or None
        Subset of time_range_valid used when searching for starting
        positions. Defaults to time_range_valid if None.
    timestep_secs : int
        Interval between frames in seconds (1800 for IMERG 30-min).
    random_seed : int or None
        Seed for the internal RNG (used in get_bin_sample shuffling).
    preselected_samples : ignored
        Reserved for future use.
    """

    def __init__(
        self, bins, patch_data, patch_index,
        sample_shape, time_range_valid, time_range_sampling=None,
        timestep_secs=5*60,
        random_seed=None, preselected_samples=None
    ):
        # Classify every patch (data + zero) into intensity bins in parallel
        binned_patches = bin_classify_patches_parallel(
            bins,
            *unpack_patches(patch_data),
            zero_value=patch_data.get("zero_value", 0),
            scale=patch_data.get("scale")
        )

        # Find all (t, i, j) positions that have a complete window
        # (no missing patches anywhere in the spatial+temporal box)
        complete_ind = indices_with_complete_sample(
            patch_index, sample_shape, time_range_valid, timestep_secs
        )

        if time_range_sampling is None:
            time_range_sampling = time_range_valid

        # For each bin, find the valid starting positions whose window
        # contains at least one patch from that bin
        self.starting_ind = [
            starting_indices_for_centers(
                p, complete_ind, sample_shape, time_range_sampling, timestep_secs
            )
            for p in binned_patches
        ]
        self.num_bins = len(self.starting_ind)
        self.rng = np.random.RandomState(seed=random_seed)
        self.preselected_samples = preselected_samples
        # Pointer tracking how far we've consumed each bin's shuffled list
        self.current_ind = np.array([len(ind) for ind in self.starting_ind])

    def get_bin_sample(self, bin_ind):
        """Return the next (t, i, j) from bin bin_ind's shuffled list.

        When the list is exhausted, it is reshuffled and restarted from
        the beginning (like a deck of cards). This ensures every position
        in a bin is visited once per "epoch" before repeating.

        Parameters
        ----------
        bin_ind : int
            Index of the intensity bin to sample from.

        Returns
        -------
        np.ndarray
            Shape (3,) — [t0, i0, j0] anchor coordinates.
        """
        patches = self.starting_ind[bin_ind]
        sample_ind = self.current_ind[bin_ind]
        if sample_ind >= patches.shape[0]:
            # Exhausted this bin's list → reshuffle and restart
            self.rng.shuffle(patches)
            sample_ind = self.current_ind[bin_ind] = 0
        else:
            self.current_ind[bin_ind] += 1
        return patches[sample_ind,:]

    def __call__(self, num):
        """Draw `num` anchor positions with equal frequency across bins.

        Parameters
        ----------
        num : int
            Number of (t0, i0, j0) samples to draw (= batch_size).

        Returns
        -------
        np.ndarray
            Shape (num, 3) — columns are [t0, i0, j0].
        """
        # Pick a random bin for each sample (uniform across num_bins)
        bins = self.rng.randint(self.num_bins, size=num)
        coords = np.stack(
            [self.get_bin_sample(b) for b in bins],
            axis=0
        )
        return coords


def bin_classify_patches(
    bins, patches, patch_coords, patch_times,
    zero_patch_coords, zero_patch_times,
    zero_value=0, metric_func=None,
    scale=None,
):
    """Classify patches into intensity bins based on a summary metric.

    The default metric is the 99th-percentile pixel value of each patch,
    which characterises the peak intensity within the patch rather than
    its mean. This is rounded for integer-type data and left as-is for
    float32 (IMERG).

    Zero patches are assigned to whichever bin contains zero_value.

    Parameters
    ----------
    bins : np.ndarray
        Monotonically increasing bin edges, shape (num_bins,).
    patches : np.ndarray
        Shape (N, H, W) — pixel data for data patches.
    patch_coords, patch_times : np.ndarray
        Coordinates and timestamps for data patches.
    zero_patch_coords, zero_patch_times : np.ndarray
        Coordinates and timestamps for zero/dry patches.
    zero_value : scalar
        Precipitation value that represents a dry patch (typically 0).
    metric_func : callable or None
        Function patch_array → 1-D metric array. Default: 99th percentile.
    scale : np.ndarray or None
        Optional look-up table to convert integer patch values to physical
        units before binning (not used for IMERG float32 data).

    Returns
    -------
    list[np.ndarray]
        Length num_bins+1. binned_patches[k] has shape (N_k, 3) where
        each row is [t, i, j] for a patch landing in bin k.
    """
    if metric_func is None:
        def metric_func(x):
            # 99th-percentile captures the peak intensity, not the mean
            xm = np.percentile(x, 99, axis=(1,2))
            if np.issubdtype(x.dtype, np.integer):
                xm = xm.round()
            return xm.astype(x.dtype)

    binned_patches = [[] for _ in range(len(bins)+1)]

    def find_bin(value):
        # bisect_left returns the index of the first bin edge >= value
        return bisect_left(bins, value)

    # All zero patches share the same metric (zero_value) → same bin
    zero_bin = find_bin(zero_value if scale is None else scale[zero_value])
    for (t,(pi,pj)) in zip(zero_patch_times, zero_patch_coords):
        binned_patches[zero_bin].append((t,pi,pj))

    # Classify each data patch by its 99th-percentile value
    patch_metrics = metric_func(patches)
    if scale is not None:
        patch_metrics = scale[patch_metrics]
    for (metric,t,(pi,pj)) in zip(patch_metrics, patch_times, patch_coords):
        patch_bin = find_bin(metric)
        binned_patches[patch_bin].append((t,pi,pj))

    # Convert lists to arrays (empty bins get a zero-row array)
    for i in range(len(binned_patches)):
        if binned_patches[i]:
            binned_patches[i] = np.array(binned_patches[i])
        else:
            binned_patches[i] = np.zeros((0,3), dtype=np.int64)

    return binned_patches


def bin_classify_patches_parallel(
    bins, patches, patch_coords, patch_times,
    zero_patch_coords, zero_patch_times,
    zero_value=0, metric_func=None,
    scale=None,
):
    """Parallel version of bin_classify_patches using Dask threads.

    Splits the patch arrays into num_cpu chunks and classifies each
    chunk in a separate thread, then merges the per-bin results.

    Parameters
    ----------
    (same as bin_classify_patches)

    Returns
    -------
    list[np.ndarray]
        Merged bin lists; same format as bin_classify_patches output.
    """
    num_patches = patches.shape[0]
    num_zeros = zero_patch_coords.shape[0]
    num_procs = multiprocessing.cpu_count()

    tasks = []
    for p in range(num_procs):
        # Divide data patches and zero patches into equal-sized chunks
        pk0 = int(round(num_patches*p/num_procs))
        pk1 = int(round(num_patches*(p+1)/num_procs))
        zk0 = int(round(num_zeros*p/num_procs))
        zk1 = int(round(num_zeros*(p+1)/num_procs))

        task = dask.delayed(bin_classify_patches)(
            bins,
            patches[pk0:pk1,...], patch_coords[pk0:pk1,...],
            patch_times[pk0:pk1],
            zero_patch_coords[zk0:zk1,...], zero_patch_times[zk0:zk1],
            zero_value=zero_value, metric_func=metric_func,
            scale=scale
        )
        tasks.append(task)

    chunked_bins = dask.compute(tasks, scheduler="threads")[0]

    # Merge per-chunk bin lists into one list per bin
    n_bins = len(chunked_bins[0])
    binned_patches = [
        np.concatenate([cb[i] for cb in chunked_bins], axis=0)
        for i in range(n_bins)
    ]
    return binned_patches


def indices_with_complete_sample(
    patch_index, sample_shape, time_range, timestep_secs
):
    """Find all (t, i, j) locations that yield samples with no missing data.

    A position (t0, i0, j0) is "complete" if, for every timestep ts in
    time_range and every patch (i0+di, j0+dj) in the spatial box, the
    key (t0 + ts*timestep_secs, i0+di, j0+dj) exists in the patch index.

    The check is performed in a Numba parallel loop over all positions
    that are already in the patch index (both data and zero patches).

    Parameters
    ----------
    patch_index : PatchIndex
        Index object whose .patch_index Numba dict is inspected.
    sample_shape : tuple[int, int]
        Spatial box size in patch units (rows, cols), e.g. (2, 2).
    time_range : tuple[int, int]
        Closed-open range of relative timestep offsets, e.g. (-16, 13).
    timestep_secs : int
        Seconds per timestep (1800 for IMERG 30-min data).

    Returns
    -------
    numba.typed.Dict
        Maps complete (t, i, j) → uint8(0). Used as a fast membership set.
    """
    ind = np.array(list(patch_index.patch_index.keys()))
    t0 = ind[:,0]
    i0 = ind[:,1]
    j0 = ind[:,2]
    n = ind.shape[0]
    complete = np.ones(n, dtype=bool)
    complete_ind = Dict.empty(
        key_type=types.UniTuple(types.int64, 3),
        value_type=types.uint8
    )

    @njit(parallel=True)
    def check_complete(index, complete, complete_ind):
        # For every existing position, verify that ALL required neighbors exist
        for k in prange(n):
            for ts in range(*time_range):
                t = t0[k] + ts*timestep_secs
                for di in range(sample_shape[0]):
                    i = i0[k] + di
                    for dj in range(sample_shape[1]):
                        j = j0[k] + dj
                        if (t,i,j) not in index:
                            complete[k] = False  # at least one neighbor is missing

        # Collect positions that passed all checks into the output set
        for k in range(n):
            if complete[k]:
                complete_ind[(t0[k],i0[k],j0[k])] = np.uint8(0)

    check_complete(patch_index.patch_index, complete, complete_ind)

    return complete_ind


def starting_indices_for_centers(
    centers, complete_ind, sample_shape, time_range, timestep_secs
):
    """Find all valid sample starting positions that contain the given center patches.

    For each patch in `centers` (a set of (t, i, j) patch locations),
    searches backwards in time and space to find all complete starting
    positions (t_start, i_start, j_start) from whose window the center
    patch would appear.

    This is how bin membership is transferred to starting positions:
    if a heavy-rain patch lands in bin 7, all starting positions that
    would include that patch in their 28-frame window are added to bin 7's
    list.

    The search uses Dask threads for outer parallelism and a Numba JIT
    function for the inner loop.

    Parameters
    ----------
    centers : np.ndarray
        Shape (K, 3) — [t, i, j] for each patch in this bin.
    complete_ind : numba.typed.Dict
        Set of complete starting positions (from indices_with_complete_sample).
    sample_shape : tuple[int, int]
        Spatial box size in patch units.
    time_range : tuple[int, int]
        Range of relative offsets to search backwards, e.g. (-1, 2).
    timestep_secs : int
        Seconds per timestep.

    Returns
    -------
    np.ndarray
        Shape (N_valid, 3) — unique valid (t_start, i_start, j_start)
        starting positions for this bin.
    """

    @njit
    def find_indices(centers, starting_ind, complete_ind):
        # For each center patch, walk backwards in time and space to find
        # all complete starting positions that would "see" this patch
        for k in range(centers.shape[0]):
            t0 = centers[k,0]
            i0 = centers[k,1]
            j0 = centers[k,2]
            for ts in range(*time_range):
                # A starting position t_start would include this patch at offset ts
                t = t0 - ts*timestep_secs
                for di in range(sample_shape[0]):
                    i = i0 - di
                    for dj in range(sample_shape[1]):
                        j = j0 - dj
                        if (t,i,j) in complete_ind:
                            starting_ind[(t,i,j)] = np.uint8(0)

    num_chunks = multiprocessing.cpu_count()

    @dask.delayed
    def chunk(i):
        # Each Dask task handles a slice of the centers array
        starting_ind = Dict.empty(
            key_type=types.UniTuple(types.int64, 3),
            value_type=types.uint8
        )
        k0 = int(round(centers.shape[0] * (i / num_chunks)))
        k1 = int(round(centers.shape[0] * ((i+1) / num_chunks)))
        find_indices(centers[k0:k1,...], starting_ind, complete_ind)
        return starting_ind

    jobs = [chunk(i) for i in range(num_chunks)]
    starting_ind = dask.compute(jobs, scheduler='threads')[0]
    # Merge all per-chunk dicts into one flat array of (t, i, j) rows
    starting_ind = np.concatenate(
        [np.array(list(st_ind.keys())) for st_ind in starting_ind if st_ind],
        axis=0
    )
    return starting_ind
