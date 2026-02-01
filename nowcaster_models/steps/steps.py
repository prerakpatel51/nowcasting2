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

    Returns
    -------
    np.ndarray
        Ensemble mean forecast (timesteps, H, W) in mm/h.
    """
    if config is None:
        config = STEPSConfig()

    # Transform to dB scale
    R_train, _ = transformation.dB_transform(
        in_precip,
        threshold=config.precip_threshold,
        zerovalue=config.zerovalue
    )

    # Set missing values with the fill value
    R_train[~np.isfinite(R_train)] = config.zerovalue

    # Estimate the motion field
    V = dense_lucaskanade(R_train)

    # The STEPS nowcast
    nowcast_method = nowcasts.get_method("steps")
    R_forecast_dB = nowcast_method(
        R_train,
        V,
        timesteps,
        n_ens_members=config.n_ens_members,
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

    # Return ensemble mean, replace NaN with 0
    result = np.nanmean(R_forecast, axis=0)
    result = np.nan_to_num(result, nan=0.0)
    return result
