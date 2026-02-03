"""STEPS nowcasting implementation."""

import numpy as np
from pysteps.utils import transformation
from pysteps import nowcasts
from pysteps.motion.lucaskanade import dense_lucaskanade

from .config import STEPSConfig


def steps(in_precip, timesteps, config=None):
    """
    Generate precipitation nowcast using STEPS algorithm.

    Parameters
    ----------
    in_precip : np.ndarray
        Input precipitation field (T, H, W) in mm/h.
    timesteps : int
        Number of forecast time steps.
    config : STEPSConfig, optional
        Configuration parameters. Uses defaults if None.
        - config.n_ensemble: Number of ensemble members to generate.
        - config.return_ensemble: If False, returns mean; if True, returns all members.

    Returns
    -------
    np.ndarray
        If return_ensemble=False: Ensemble mean forecast (timesteps, H, W) in mm/h.
        If return_ensemble=True: All ensemble forecasts (n_ensemble, timesteps, H, W) in mm/h.
        Returns zeros if input is unsuitable for STEPS (dry, low variance, etc.)
    """
    if config is None:
        config = STEPSConfig()

    H, W = in_precip.shape[1], in_precip.shape[2]
    n_ensemble = config.n_ensemble
    return_ensemble = config.return_ensemble

    # Output shape for zero-fill fallback
    if return_ensemble:
        output_shape = (n_ensemble, timesteps, H, W)
    else:
        output_shape = (timesteps, H, W)

    # Check 1: Dry input - if max value below threshold, return zeros
    if np.max(in_precip) < config.precip_threshold:
        return np.zeros(output_shape, dtype=np.float32)

    # Check 2: Low variance input - STEPS needs texture for motion estimation
    # If variance is too low, noise generator and motion estimation will fail
    if np.var(in_precip) < 1e-6:
        return np.zeros(output_shape, dtype=np.float32)

    try:
        # Transform to dB scale
        R_train, _ = transformation.dB_transform(
            in_precip,
            threshold=config.precip_threshold,
            zerovalue=config.zerovalue
        )

        # Set non-finite values to zerovalue
        R_train[~np.isfinite(R_train)] = config.zerovalue

        # Check 3: After transform, verify we have valid data
        if not np.any(R_train > config.zerovalue):
            return np.zeros(output_shape, dtype=np.float32)

        # Estimate the motion field
        V = dense_lucaskanade(R_train)

        # Check 4: Handle motion field issues
        if V is None:
            return np.zeros(output_shape, dtype=np.float32)

        # Replace non-finite values with zero (no motion at those pixels)
        V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

        # Run STEPS nowcast with specified ensemble members
        nowcast_method = nowcasts.get_method("steps")
        R_forecast_dB = nowcast_method(
            R_train,
            V,
            timesteps,
            n_ens_members=n_ensemble,
            n_cascade_levels=config.n_cascade_levels,
            precip_thr=config.precip_thr,
            kmperpixel=config.kmperpixel,
            timestep=config.timestep
        )

        # Back-transform to rain rates (mm/h)
        R_forecast, _ = transformation.dB_transform(
            R_forecast_dB,
            threshold=config.precip_thr,
            inverse=True
        )

        # Clean up non-finite values
        R_forecast = np.nan_to_num(R_forecast, nan=0.0, posinf=0.0, neginf=0.0)
        R_forecast = np.maximum(R_forecast, 0.0)  # Ensure non-negative

        if return_ensemble:
            # Return all ensemble members: (n_ensemble, timesteps, H, W)
            return R_forecast.astype(np.float32)
        else:
            # Return ensemble mean: (timesteps, H, W)
            result = np.mean(R_forecast, axis=0)
            return result.astype(np.float32)

    except Exception:
        # If any step fails, return zeros
        # Common failures: singular matrix, zero-size array, non-finite values
        return np.zeros(output_shape, dtype=np.float32)
