"""Data transformations for converting raw IMERG values to model inputs.

Copied from ldcast/features/transform.py — only the functions needed for
IMERG data (normalize_threshold with log10 + standardization). Scale-table
functions (scale_log_norm, scale_norm) are not needed since IMERG stores
float32 precipitation directly (not quantized integers).

Transform pipeline for IMERG
-----------------------------
Raw IMERG values are in mm/hr (float32, range roughly 0–100).
The model expects normalized values centred near 0 with unit variance.
Three operations are applied in sequence:

  1. Threshold clamp: x < 0.1  →  x = 0.01
     Prevents log10(0) = -inf. Values below 0.1 mm/hr are treated as
     effectively dry.

  2. Log10:  x = log10(x)
     Compresses the heavy tail of the precipitation distribution.
     After clamping, range is approximately log10(0.01)=-2 to log10(100)=2.

  3. Standardize: x = (x - mean) / std
     Centers the distribution. mean=-0.1399, std=0.4151 were computed
     from the IMERG training set.

All inner kernels (normalize_array, normalize_threshold_array) are compiled
by Numba with parallel=True so they run across all available CPU cores.
"""

import concurrent.futures
import multiprocessing

from numba import njit, prange
import numpy as np
from scipy.ndimage import convolve


def quick_cast(x, y):
    """Multi-threaded array dtype casting using a thread pool.

    Splits the first axis of x into num_cpu chunks and casts each chunk
    in a separate thread. Useful when converting large float32 arrays to
    a lower-precision dtype (e.g. float16) without blocking on the GIL.

    Parameters
    ----------
    x : np.ndarray
        Source array to cast.
    y : np.ndarray
        Destination array with the target dtype (modified in place).
    """
    num_threads = multiprocessing.cpu_count()
    with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
        futures = {}
        limits = np.linspace(0, x.shape[0], num_threads+1).round().astype(int)
        def _cast(k0, k1):
            y[k0:k1,...] = x[k0:k1,...]
        for k in range(len(limits)-1):
            args = (_cast, limits[k], limits[k+1])
            futures[executor.submit(*args)] = k
        concurrent.futures.wait(futures)


def normalize(mean=0.0, std=1.0, dtype=np.float32):
    """Create a standard normalization transform: (x - mean) / std.

    Returns a stateful closure that pre-allocates an output buffer on the
    first call and reuses it for subsequent calls of the same shape,
    avoiding repeated memory allocation during training.

    Parameters
    ----------
    mean : float
        Dataset mean to subtract.
    std : float
        Dataset standard deviation to divide by.
    dtype : np.dtype
        Output dtype. If float32, returns the intermediate buffer directly.
        Otherwise uses quick_cast to convert.

    Returns
    -------
    callable
        transform(raw: np.ndarray) → np.ndarray (same shape, dtype=dtype)
    """
    scaled = scaled_dt = None

    def transform(raw):
        nonlocal scaled, scaled_dt
        # Reallocate buffer only when the input shape changes
        if (scaled is None) or (scaled.shape != raw.shape):
            scaled = np.empty_like(raw, dtype=np.float32)
            scaled_dt = np.empty_like(raw, dtype=dtype)
        normalize_array(raw, scaled, mean, std)

        if dtype == np.float32:
            return scaled
        else:
            quick_cast(scaled, scaled_dt)
            return scaled_dt

    return transform


def normalize_threshold(mean=0.0, std=1.0, threshold=0.0, fill_value=0.0, log=False):
    """Create a normalization transform with thresholding and optional log.

    This is the primary transform used for IMERG precipitation data.
    Returns a stateful closure with a pre-allocated output buffer.

    Transform sequence applied to each element:
        1. If x < threshold  →  x = fill_value
        2. If log=True       →  x = log10(x)
        3.                      x = (x - mean) / std

    The inner computation is done by normalize_threshold_array, a Numba
    @njit(parallel=True) kernel that processes elements across CPU cores.

    Parameters
    ----------
    mean : float
        Dataset mean in log10 space (e.g. -0.1399 for IMERG).
    std : float
        Dataset std in log10 space (e.g. 0.4151 for IMERG).
    threshold : float
        Values below this are replaced with fill_value (e.g. 0.1 mm/hr).
    fill_value : float
        Replacement for sub-threshold values (e.g. 0.01 mm/hr).
    log : bool
        If True, apply log10 after clamping and before standardizing.

    Returns
    -------
    callable
        transform(raw: np.ndarray) → np.ndarray float32, same shape.
    """
    scaled = None

    def transform(raw):
        nonlocal scaled
        # Reallocate only when input shape changes (avoids per-batch malloc)
        if (scaled is None) or (scaled.shape != raw.shape):
            scaled = np.empty_like(raw, dtype=np.float32)
        normalize_threshold_array(raw, scaled, mean, std, threshold, fill_value, log=log)
        return scaled

    return transform


def combine(transforms, memory_format="channels_first", dim=3):
    """Create a transform that applies multiple transforms and concatenates.

    Useful when a model input is built from several sources (e.g. radar +
    NWP) that each have their own normalization. Outputs are concatenated
    along the channel axis.

    Parameters
    ----------
    transforms : list[callable]
        One transform per source array; applied element-wise to *raw args.
    memory_format : str
        "channels_first" (default) → concatenate along axis 1.
        "channels_last" → concatenate along axis -1.
    dim : int
        Spatial dimensionality of the data (e.g. 3 for (B, T, H, W)).

    Returns
    -------
    callable
        transform(*raw) → np.ndarray with an extra channel axis.
    """
    channels_axis = 1 if (memory_format == "channels_first") else -1

    def transform(*raw):
        transformed = [t(r) for (t, r) in zip(transforms, raw)]
        for i in range(len(transformed)):
            # Add a channel dimension if not already present
            if transformed[i].ndim == dim + 1:
                transformed[i] = np.expand_dims(transformed[i], channels_axis)
        return np.concatenate(transformed, axis=channels_axis)

    return transform


class Antialiasing:
    """Gaussian smoothing filter to reduce quantization aliasing artifacts.

    Applies a 5×5 Gaussian kernel (sigma=0.5) to each frame of a
    (B, T, H, W) array. Edge effects are corrected by dividing by the
    convolution of the kernel with a ones array at the same spatial size.

    Smoothing is parallelised across (batch, time) frames using a
    persistent thread pool so the pool overhead is paid only once.

    Note: Not used for IMERG data (which is stored as float32 with no
    quantization). Retained for compatibility with radar/MCH pipelines.
    """

    def __init__(self):
        # Build a 5×5 Gaussian kernel with sigma=0.5
        (x, y) = np.mgrid[-2:3,-2:3]
        self.kernel = np.exp(-0.5*(x**2+y**2)/(0.5**2))
        self.kernel /= self.kernel.sum()
        # Cache edge-correction factors per spatial shape
        self.edge_factors = {}
        # Cache smoothed output buffers per array shape
        self.img_smooth = {}
        num_threads = multiprocessing.cpu_count()
        # Persistent thread pool to avoid re-creation overhead per call
        self.executor = concurrent.futures.ThreadPoolExecutor(num_threads)

    def __call__(self, img):
        """Apply Gaussian smoothing to all frames in img.

        Parameters
        ----------
        img : np.ndarray
            Shape (B, T, H, W) — input frames (any float dtype).

        Returns
        -------
        np.ndarray
            Same shape as img, Gaussian-smoothed with edge correction.
        """
        img_shape = img.shape[-2:]

        # Compute and cache edge-correction factor for this spatial size
        if img_shape not in self.edge_factors:
            s = convolve(np.ones(img_shape, dtype=np.float32),
                self.kernel, mode="constant")
            s = 1.0/s
            self.edge_factors[img_shape] = s
        else:
            s = self.edge_factors[img_shape]

        # Allocate (or reuse) the output buffer for this array shape
        if img.shape not in self.img_smooth:
            img_smooth = np.empty_like(img)
            self.img_smooth[img_shape] = img_smooth
        else:
            img_smooth = self.img_smooth[img_shape]

        def _convolve_frame(i, j):
            """Convolve a single (H, W) frame and apply edge correction."""
            convolve(img[i,j,:,:], self.kernel,
                mode="constant", output=img_smooth[i,j,:,:])
            img_smooth[i,j,:,:] *= s

        # Submit one task per (batch, time) frame to the thread pool
        futures = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                args = (_convolve_frame, i, j)
                futures.append(self.executor.submit(*args))
        concurrent.futures.wait(futures)

        return img_smooth


@njit(parallel=True)
def normalize_array(in_arr, out_arr, mean, std):
    """Numba kernel for element-wise normalization: out = (in - mean) / std.

    Flattens both arrays to 1-D and processes them in parallel with prange.
    Called by the normalize() closure.

    Parameters
    ----------
    in_arr : np.ndarray
        Input array (any shape; ravelled internally).
    out_arr : np.ndarray
        Pre-allocated float32 output (same shape as in_arr; modified in place).
    mean : float
        Value to subtract from each element.
    std : float
        Value to divide each element by (inverted to multiplication for speed).
    """
    mean = np.float32(mean)
    inv_std = np.float32(1.0/std)
    in_arr = in_arr.ravel()
    out_arr = out_arr.ravel()
    for i in prange(in_arr.shape[0]):
        out_arr[i] = (in_arr[i]-mean)*inv_std


@njit(parallel=True)
def normalize_threshold_array(
    in_arr, out_arr,
    mean, std,
    threshold, fill_value, log=False
):
    """Numba kernel: threshold clamp → optional log10 → normalize.

    For each element:
        1. If value < threshold, replace with fill_value
        2. Optionally apply log10
        3. Normalize: (x - mean) / std

    Runs in parallel across all CPU cores via prange. Called by the
    normalize_threshold() closure once per batch.

    Parameters
    ----------
    in_arr : np.ndarray
        Raw input array (any shape; ravelled internally).
    out_arr : np.ndarray
        Pre-allocated float32 output (same shape; modified in place).
    mean : float
        Dataset mean in transformed space.
    std : float
        Dataset std in transformed space.
    threshold : float
        Values below this are clamped to fill_value (e.g. 0.1 mm/hr).
    fill_value : float
        Replacement for sub-threshold values (e.g. 0.01 mm/hr).
    log : bool
        If True, apply log10 after clamping and before standardizing.
    """
    mean = np.float32(mean)
    inv_std = np.float32(1.0/std)
    threshold = np.float32(threshold)
    fill_value = np.float32(fill_value)
    in_arr = in_arr.ravel()
    out_arr = out_arr.ravel()
    for i in prange(in_arr.shape[0]):
        x = in_arr[i]
        # Step 1: clamp sub-threshold values
        if x < threshold:
            x = fill_value
        # Step 2: optional log10 compression
        if log:
            x = np.log10(x)
        # Step 3: standardize
        out_arr[i] = (x-mean)*inv_std
