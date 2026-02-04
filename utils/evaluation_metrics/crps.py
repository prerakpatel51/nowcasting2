"""
Continuous Ranked Probability Score (CRPS) for Ensemble Forecasts
=================================================================

This module provides functions for computing the Continuous Ranked Probability
Score (CRPS), a strictly proper scoring rule widely used in meteorology for
evaluating probabilistic forecasts.

The CRPS measures the difference between a predicted cumulative distribution F
and an observation y:

    CRPS(F, y) = integral over R of (F(x) - H(x - y))^2 dx

where H is the Heaviside step function.

For ensemble forecasts, we use the energy form which is equivalent and
amenable to Monte Carlo estimation:

    CRPS(F, y) = E[|X - y|] - 0.5 * E[|X - X'|]

where X and X' are independent samples from F (i.e., ensemble members).

Key Properties:
    - CRPS is a strictly proper scoring rule
    - Lower CRPS indicates better forecast skill
    - For a deterministic forecast (single ensemble member), CRPS equals MAE
    - CRPS generalizes MAE to probabilistic forecasts

References:
    [1] Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
        prediction, and estimation. JASA, 102(477), 359-378.
    [2] Hersbach, H. (2000). Decomposition of the continuous ranked probability
        score for ensemble prediction systems. Weather and Forecasting, 15(5).
"""

import os
import numpy as np
import h5py
from typing import Tuple, Optional


def crps_ensemble(
    ensemble_forecasts: np.ndarray,
    observations: np.ndarray,
    axis: int = 0
) -> np.ndarray:
    """
    Compute the Continuous Ranked Probability Score (CRPS) for ensemble forecasts.

    Uses the energy form of CRPS which is efficient for ensemble-based estimation:

        CRPS(F, y) = E[|X - y|] - 0.5 * E[|X - X'|]

    where X and X' are independent samples (ensemble members) from the forecast
    distribution F, and y is the observation.

    Parameters
    ----------
    ensemble_forecasts : np.ndarray
        Array of ensemble forecast values. The ensemble dimension is specified
        by the `axis` parameter. Shape: (..., n_ensemble, ...) where n_ensemble
        is along the specified axis.
    observations : np.ndarray
        Array of observed values. Shape should match ensemble_forecasts with
        the ensemble dimension removed.
    axis : int, default=0
        The axis along which ensemble members are stored.

    Returns
    -------
    np.ndarray
        CRPS values with the same shape as observations. Lower values indicate
        better forecast skill.

    Examples
    --------
    >>> # 5 ensemble members, single observation
    >>> ensemble = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> obs = 2.5
    >>> crps = crps_ensemble(ensemble, obs, axis=0)

    >>> # Shape (n_ensemble, time, height, width) -> CRPS shape (time, height, width)
    >>> ensemble = np.random.rand(10, 12, 64, 64)  # 10 members
    >>> obs = np.random.rand(12, 64, 64)
    >>> crps = crps_ensemble(ensemble, obs, axis=0)

    Notes
    -----
    The energy form computes:
    - Term 1: Mean absolute difference between each ensemble member and observation
    - Term 2: Mean absolute difference between all pairs of ensemble members

    For n ensemble members, Term 2 requires O(n^2) comparisons, but we use
    an efficient implementation that avoids explicit loops.
    """
    # Move ensemble axis to the front for easier computation
    ensemble = np.moveaxis(ensemble_forecasts, axis, 0)
    n_ensemble = ensemble.shape[0]

    # Ensure observations have compatible shape (no ensemble dimension)
    obs = np.asarray(observations)

    # Term 1: E[|X - y|] - mean absolute error between ensemble and observation
    # Shape: (n_ensemble, ...) -> mean over axis 0 -> (...)
    term1 = np.mean(np.abs(ensemble - obs), axis=0)

    # Term 2: E[|X - X'|] - mean absolute difference between ensemble pairs
    # Efficient computation: for each member, compute mean abs diff to all others
    # This is equivalent to summing |X_i - X_j| for all i,j and dividing by n^2
    term2 = 0.0
    for i in range(n_ensemble):
        for j in range(i + 1, n_ensemble):
            term2 += np.abs(ensemble[i] - ensemble[j])

    # Normalize: we computed sum for i < j, but need mean over all pairs
    # Total pairs = n_ensemble^2, unique pairs (i < j) = n_ensemble * (n_ensemble - 1) / 2
    # E[|X - X'|] = 2 * sum_{i<j} |X_i - X_j| / (n_ensemble * (n_ensemble - 1))
    # But in energy form, we need E[|X - X'|] including diagonal (which is 0)
    # So: E[|X - X'|] = 2 * sum_{i<j} |X_i - X_j| / n_ensemble^2
    if n_ensemble > 1:
        term2 = 2.0 * term2 / (n_ensemble * n_ensemble)
    else:
        term2 = 0.0

    # CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    crps = term1 - 0.5 * term2

    return crps


def evaluate_crps_from_h5(
    input_file: str,
    output_dir: Optional[str] = None,
    batch_size: int = 100
) -> Tuple[np.ndarray, dict]:
    """
    Evaluate CRPS from ensemble forecasts stored in an HDF5 file.

    Reads predictions and targets from an HDF5 file created by run_nowcast.py,
    computes CRPS for each sample, and saves the results.

    Parameters
    ----------
    input_file : str
        Path to the HDF5 file containing 'predictions' and 'targets' datasets.
        Expected format:
        - Ensemble mode: predictions (N, E, T, H, W), targets (N, T, H, W)
        - Mean mode: Not supported (requires ensemble for CRPS)
    output_dir : str, optional
        Directory to save results. If None, saves in the same directory as
        input_file.
    batch_size : int, default=100
        Number of samples to process at a time to manage memory.

    Returns
    -------
    crps_results : np.ndarray
        Array of CRPS values with shape (N, T, H, W) - CRPS for each
        sample, timestep, and spatial location.
    summary : dict
        Summary statistics including:
        - 'mean_crps': Overall mean CRPS
        - 'mean_crps_per_timestep': Mean CRPS for each forecast timestep
        - 'std_crps': Standard deviation of CRPS
        - 'n_samples': Number of samples evaluated
        - 'n_ensemble': Number of ensemble members
        - 'output_file': Path to saved results

    Raises
    ------
    ValueError
        If the input file is not in ensemble mode (CRPS requires ensemble).

    Examples
    --------
    >>> # Evaluate CRPS from STEPS ensemble forecast
    >>> results, summary = evaluate_crps_from_h5(
    ...     'nowcaster_results/forecast_results/steps_ensemble_20/test/results.h5'
    ... )
    >>> print(f"Mean CRPS: {summary['mean_crps']:.4f}")

    Notes
    -----
    The function saves two output files:
    1. crps_values.h5: Full CRPS array with shape (N, T, H, W)
    2. crps_summary.npz: Summary statistics and per-timestep metrics
    """
    # Validate input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    os.makedirs(output_dir, exist_ok=True)

    # Read file metadata and validate
    with h5py.File(input_file, 'r') as f:
        ensemble_mode = f.attrs.get('ensemble_mode', False)
        if not ensemble_mode:
            raise ValueError(
                "CRPS requires ensemble forecasts. The input file is not in "
                "ensemble mode. Please run the forecast with return_ensemble=True."
            )

        n_ensemble = f.attrs.get('n_ensemble', None)
        pred_shape = f['predictions'].shape  # (N, E, T, H, W)
        target_shape = f['targets'].shape    # (N, T, H, W)

        n_samples = pred_shape[0]
        n_timesteps = pred_shape[2]
        height, width = pred_shape[3], pred_shape[4]

    print("=" * 60)
    print("CRPS Evaluation")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Predictions shape: {pred_shape}")
    print(f"Targets shape: {target_shape}")
    print(f"Number of samples: {n_samples}")
    print(f"Number of ensemble members: {n_ensemble}")
    print(f"Forecast timesteps: {n_timesteps}")
    print(f"Spatial dimensions: {height} x {width}")
    print("=" * 60)

    # Initialize output array
    crps_all = np.zeros((n_samples, n_timesteps, height, width), dtype=np.float32)

    # Process in batches to manage memory
    n_batches = (n_samples + batch_size - 1) // batch_size

    with h5py.File(input_file, 'r') as f:
        predictions = f['predictions']
        targets = f['targets']

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)

            # Load batch data
            # predictions: (batch, E, T, H, W)
            # targets: (batch, T, H, W)
            pred_batch = predictions[start_idx:end_idx]
            target_batch = targets[start_idx:end_idx]

            # Compute CRPS for each sample in batch
            for i in range(pred_batch.shape[0]):
                sample_idx = start_idx + i
                # pred_batch[i] has shape (E, T, H, W)
                # target_batch[i] has shape (T, H, W)
                crps_all[sample_idx] = crps_ensemble(
                    pred_batch[i],
                    target_batch[i],
                    axis=0  # Ensemble axis
                )

            # Progress update
            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                print(f"Processed {end_idx}/{n_samples} samples...")

    # Compute summary statistics
    mean_crps = np.mean(crps_all)
    std_crps = np.std(crps_all)
    mean_crps_per_timestep = np.mean(crps_all, axis=(0, 2, 3))  # (T,)
    mean_crps_spatial = np.mean(crps_all, axis=(0, 1))  # (H, W)

    # Create summary dict
    summary = {
        'mean_crps': float(mean_crps),
        'std_crps': float(std_crps),
        'mean_crps_per_timestep': mean_crps_per_timestep,
        'mean_crps_spatial': mean_crps_spatial,
        'n_samples': n_samples,
        'n_ensemble': n_ensemble,
        'n_timesteps': n_timesteps,
        'input_file': input_file,
    }

    # Save CRPS values to HDF5
    crps_output_file = os.path.join(output_dir, 'crps_values.h5')
    with h5py.File(crps_output_file, 'w') as f:
        f.create_dataset('crps', data=crps_all, compression='gzip')
        f.attrs['n_samples'] = n_samples
        f.attrs['n_ensemble'] = n_ensemble
        f.attrs['n_timesteps'] = n_timesteps
        f.attrs['mean_crps'] = mean_crps
        f.attrs['std_crps'] = std_crps

    # Save summary statistics to npz
    summary_output_file = os.path.join(output_dir, 'crps_summary.npz')
    np.savez(
        summary_output_file,
        mean_crps=mean_crps,
        std_crps=std_crps,
        mean_crps_per_timestep=mean_crps_per_timestep,
        mean_crps_spatial=mean_crps_spatial,
        n_samples=n_samples,
        n_ensemble=n_ensemble,
        n_timesteps=n_timesteps
    )

    summary['crps_output_file'] = crps_output_file
    summary['summary_output_file'] = summary_output_file

    # Print results
    print("\n" + "=" * 60)
    print("CRPS Evaluation Results")
    print("=" * 60)
    print(f"Mean CRPS: {mean_crps:.6f}")
    print(f"Std CRPS:  {std_crps:.6f}")
    print("\nCRPS per forecast timestep:")
    for t, crps_t in enumerate(mean_crps_per_timestep):
        print(f"  t+{t+1}: {crps_t:.6f}")
    print(f"\nResults saved to:")
    print(f"  - CRPS values: {crps_output_file}")
    print(f"  - Summary: {summary_output_file}")
    print("=" * 60)

    return crps_all, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate CRPS from ensemble forecast HDF5 files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate CRPS from a single file
  python crps.py --input forecast_results/steps_ensemble_20/test/results.h5
        """
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="Path to HDF5 file with ensemble predictions"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="Directory to save results (default: same as input)"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help="Batch size for processing (default: 100)"
    )

    args = parser.parse_args()

    evaluate_crps_from_h5(
        args.input,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
