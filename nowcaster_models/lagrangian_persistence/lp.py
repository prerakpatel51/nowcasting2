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
        Returns zeros if input is unsuitable (dry, low variance, etc.)
    """
    if config is None:
        config = LPConfig()

    # Output shape for zero-fill fallback
    output_shape = (timesteps, in_precip.shape[1], in_precip.shape[2])

    # Check 1: Dry input - if max value below threshold, return zeros
    if np.max(in_precip) < config.precip_threshold:
        return np.zeros(output_shape, dtype=np.float32)

    # Check 2: Low variance input - need texture for motion estimation
    if np.var(in_precip) < 1e-6:
        return np.zeros(output_shape, dtype=np.float32)

    try:
        # Transform to dB scale
        R_train, _ = transformation.dB_transform(
            in_precip,
            threshold=config.precip_threshold,
            zerovalue=config.zerovalue
        )

        # Handle non-finite values
        R_train[~np.isfinite(R_train)] = config.zerovalue

        # Check 3: After transform, verify we have valid data
        if not np.any(R_train > config.zerovalue):
            return np.zeros(output_shape, dtype=np.float32)

        # Estimate the motion field with Lucas-Kanade
        V = dense_lucaskanade(R_train)

        # Check 4: Handle motion field issues
        if V is None:
            return np.zeros(output_shape, dtype=np.float32)

        # Replace non-finite values with zero (no motion at those pixels)
        V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

        # Extrapolate the last observation using the motion field
        extrapolate = nowcasts.get_method("extrapolation")
        R_f = extrapolate(R_train[-1, :, :], V, timesteps)

        # Back-transform to rain rate (mm/h)
        R_f, _ = transformation.dB_transform(
            R_f,
            threshold=config.precip_threshold,
            inverse=True
        )

        # Clean up output
        R_f = np.nan_to_num(R_f, nan=0.0, posinf=0.0, neginf=0.0)
        R_f = np.maximum(R_f, 0.0)  # Ensure non-negative

        return R_f.astype(np.float32)

    except Exception:
        # If any step fails, return zeros
        # Common failures: singular matrix, zero-size array, non-finite values
        return np.zeros(output_shape, dtype=np.float32)
