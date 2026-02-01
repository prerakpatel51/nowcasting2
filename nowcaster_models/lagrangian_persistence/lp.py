"""Lagrangian Persistence nowcasting implementation."""

import numpy as np
from pysteps.utils import transformation
from pysteps import nowcasts
from pysteps.motion.lucaskanade import dense_lucaskanade

from .config import LPConfig


def lagrangian_persistence(in_precip, timesteps, config=None):
    """
    Generate precipitation nowcast using Lagrangian Persistence.

    Lagrangian persistence extrapolates the last observed precipitation field
    along the estimated motion vectors (advection-based persistence).

    Parameters
    ----------
    in_precip : np.ndarray
        Input precipitation field (T, H, W) in mm/h.
    timesteps : int
        Number of forecast time steps.
    config : LPConfig, optional
        Configuration parameters. Uses defaults if None.

    Returns
    -------
    np.ndarray
        Forecast precipitation (timesteps, H, W) in mm/h.
        Guaranteed to have no NaN values.
    """
    if config is None:
        config = LPConfig()

    # Transform to dB scale
    R_train, _ = transformation.dB_transform(
        in_precip,
        threshold=config.precip_threshold,
        zerovalue=config.zerovalue
    )

    # Handle non-finite values BEFORE motion estimation
    R_train[~np.isfinite(R_train)] = config.zerovalue

    # Estimate the motion field with Lucas-Kanade
    V = dense_lucaskanade(R_train)

    # Extrapolate the last observation using the motion field
    extrapolate = nowcasts.get_method("extrapolation")
    R_f = extrapolate(R_train[-1, :, :], V, timesteps)

    # Back-transform to rain rate (mm/h)
    # Use the same threshold as the forward transform for consistency
    R_f, _ = transformation.dB_transform(
        R_f,
        threshold=config.precip_threshold,
        inverse=True
    )

    # Ensure no NaN values in output - replace with 0 (no rain)
    R_f = np.nan_to_num(R_f, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure non-negative values (rain rate cannot be negative)
    R_f = np.maximum(R_f, 0.0)

    return R_f
