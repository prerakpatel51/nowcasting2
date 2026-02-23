"""Patch storage and loading for IMERG data.

Copied from ldcast/features/patches.py — only the save/load/unpack functions
needed for the IMERG pipeline (no MCH/DWD/IFS/COSMO-specific extraction).

Patch file format
-----------------
Each yearly NetCDF4 file stores two categories of patches:

  Data patches  — patches where precipitation > threshold.
                  Pixel arrays are stored in full (N, H, W).
  Zero patches  — dry/near-zero patches.
                  Only (row, col) coordinates and timestamps are stored;
                  no pixel data, saving significant disk space.

Both categories share the same coordinate/time format so that the downstream
PatchIndex can handle them uniformly (zero patches get a sentinel index).
"""

import os

import dask
import netCDF4
import numpy as np


def save_patches(patch_data, patch_coords, patch_times,
    zero_patch_coords, zero_patch_times, out_fn, zero_value=0, scale=None):
    """Save extracted patches to a compressed NetCDF4 file.

    Uses zlib compression (level 1) with chunk-aligned storage for the
    patch array so that sequential reads are efficient.

    Parameters
    ----------
    patch_data : np.ndarray
        Shape (N, H, W) — pixel values for non-zero patches.
    patch_coords : np.ndarray
        Shape (N, 2) — (row, col) patch grid indices for data patches.
    patch_times : np.ndarray
        Shape (N,) — Unix timestamps (int64 seconds) for data patches.
    zero_patch_coords : np.ndarray
        Shape (M, 2) — grid indices for zero/dry patches.
    zero_patch_times : np.ndarray
        Shape (M,) — Unix timestamps for zero patches.
    out_fn : str
        Output file path (created or overwritten).
    zero_value : scalar
        Scalar stored as a file-level attribute; used by the downstream
        pipeline to fill zero patches in assembled frames.
    scale : np.ndarray or None
        Optional look-up table for integer-quantized data (not used for
        IMERG float32). If provided, stored as a 1-D variable.
    """
    with netCDF4.Dataset(out_fn, 'w') as ds:
        # Define dimensions
        ds.createDimension("dim_patch", patch_data.shape[0])
        ds.createDimension("dim_zero_patch", zero_patch_coords.shape[0])
        ds.createDimension("dim_coord", 2)
        ds.createDimension("dim_height", patch_data.shape[1])
        ds.createDimension("dim_width", patch_data.shape[2])

        # All variables use zlib compression (level 1 — fast + good ratio)
        var_args = {"zlib": True, "complevel": 1}

        # Chunk the patch array so reading a subset of patches is efficient
        chunksizes = (min(2**10, patch_data.shape[0]), patch_data.shape[1], patch_data.shape[2])
        var_patch = ds.createVariable("patches", patch_data.dtype,
            ("dim_patch","dim_height","dim_width"), chunksizes=chunksizes, **var_args)
        var_patch[:] = patch_data

        var_patch_coord = ds.createVariable("patch_coords", patch_coords.dtype,
            ("dim_patch","dim_coord"), **var_args)
        var_patch_coord[:] = patch_coords

        var_patch_time = ds.createVariable("patch_times", patch_times.dtype,
            ("dim_patch",), **var_args)
        var_patch_time[:] = patch_times

        var_zero_patch_coord = ds.createVariable("zero_patch_coords", zero_patch_coords.dtype,
            ("dim_zero_patch","dim_coord"), **var_args)
        var_zero_patch_coord[:] = zero_patch_coords

        var_zero_patch_time = ds.createVariable("zero_patch_times", zero_patch_times.dtype,
            ("dim_zero_patch",), **var_args)
        var_zero_patch_time[:] = zero_patch_times

        # Store zero_value as a file attribute so readers know the fill value
        ds.zero_value = zero_value

        # Optional scale table for integer-quantized variables
        if scale is not None:
            ds.createDimension("dim_scale", len(scale))
            var_scale = ds.createVariable("scale", scale.dtype, ("dim_scale",), **var_args)
            var_scale[:] = scale


def load_patches(fn, in_memory=True):
    """Load a single NetCDF patch file into a dictionary.

    By default reads the entire file into a bytes buffer before parsing.
    This avoids repeated disk seeks during NetCDF variable access and is
    faster for files loaded in parallel Dask tasks.

    Parameters
    ----------
    fn : str
        Path to the NetCDF4 file.
    in_memory : bool
        If True (default), read the raw bytes into memory first and parse
        from the buffer. If False, stream directly from disk.

    Returns
    -------
    dict with keys:
        "patches"           : np.ndarray  (N, H, W)  float32
        "patch_coords"      : np.ndarray  (N, 2)     uint16
        "patch_times"       : np.ndarray  (N,)       int64
        "zero_patch_coords" : np.ndarray  (M, 2)     uint16
        "zero_patch_times"  : np.ndarray  (M,)       int64
        "zero_value"        : scalar      (file-level attribute)
        "scale"             : np.ndarray  (optional, if present in file)
    """
    if in_memory:
        # Read file into a bytes buffer; faster for parallel Dask loads
        with open(fn, 'rb') as f:
            ds_raw = f.read()
        fn = None   # netCDF4.Dataset accepts fn=None when memory is provided
    else:
        ds_raw = None

    with netCDF4.Dataset(fn, 'r', memory=ds_raw) as ds:
        patch_data = {
            "patches":           np.array(ds["patches"]),
            "patch_coords":      np.array(ds["patch_coords"]),
            "patch_times":       np.array(ds["patch_times"]),
            "zero_patch_coords": np.array(ds["zero_patch_coords"]),
            "zero_patch_times":  np.array(ds["zero_patch_times"]),
            "zero_value":        ds.zero_value
        }
        if "scale" in ds.variables:
            patch_data["scale"] = np.array(ds["scale"])

    return patch_data


def load_all_patches(patch_dir, var):
    """Load and concatenate patch data from all NetCDF files for a variable.

    Scans `patch_dir` for files whose second underscore-delimited field
    matches `var` (e.g. "patches_IMERG_2015.nc" → var "IMERG").
    Files are loaded in parallel using Dask delayed tasks with
    `scheduler="processes"` so that decompression and NumPy conversion
    run in separate OS processes (bypassing the GIL).

    All yearly arrays are concatenated along the first axis so the caller
    gets one flat dictionary spanning the entire dataset.

    Parameters
    ----------
    patch_dir : str
        Directory containing patch NetCDF files.
    var : str
        Variable name to filter for (e.g. "IMERG").

    Returns
    -------
    dict
        Concatenated patch data with keys:
        patches, patch_coords, patch_times, zero_patch_coords,
        zero_patch_times (and optionally zero_value, scale).
    """
    files = os.listdir(patch_dir)
    jobs = []
    for fn in files:
        # File naming convention: patches_{var}_{year}.nc
        file_var = fn.split("_")[1]
        if file_var == var:
            fn = os.path.join(patch_dir, fn)
            jobs.append(dask.delayed(load_patches)(fn))

    # Load all files in parallel across processes
    file_data = dask.compute(jobs, scheduler="processes")[0]

    # Concatenate arrays from all yearly files along the patch axis
    patch_data = {}
    keys = ["patches", "patch_coords", "patch_times",
        "zero_patch_coords", "zero_patch_times"]
    for k in keys:
        patch_data[k] = np.concatenate(
            [fd[k] for fd in file_data],
            axis=0
        )
    # Metadata fields are the same across files; take from first file
    patch_data["zero_value"] = file_data[0]["zero_value"]
    if "scale" in file_data[0]:
        patch_data["scale"] = file_data[0]["scale"]

    return patch_data


def unpack_patches(patch_data):
    """Unpack a patch data dictionary into its five component arrays.

    Convenience function used by PatchIndex and EqualFrequencySampler
    to avoid repeating the same dict key lookups.

    Parameters
    ----------
    patch_data : dict
        Patch dictionary with standard keys.

    Returns
    -------
    tuple
        (patches, patch_coords, patch_times,
         zero_patch_coords, zero_patch_times)
    """
    return (
        patch_data["patches"],
        patch_data["patch_coords"],
        patch_data["patch_times"],
        patch_data["zero_patch_coords"],
        patch_data["zero_patch_times"]
    )
