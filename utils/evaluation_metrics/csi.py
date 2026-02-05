"""
Critical Success Index (CSI) for Deterministic and Ensemble Forecasts
=====================================================================

This module provides functions for computing the Critical Success Index (CSI),
also known as the Threat Score, a verification metric widely used in
meteorology for evaluating precipitation forecasts at fixed intensity
thresholds.

CSI is defined as:

    CSI = TP / (TP + FP + FN)

where:
    TP (True Positives)  = both forecast and observation exceed the threshold
    FP (False Positives) = forecast exceeds threshold but observation does not
    FN (False Negatives) = observation exceeds threshold but forecast does not

Key Properties:
    - CSI ranges from 0 (no skill) to 1 (perfect forecast)
    - CSI is sensitive to both false alarms and misses
    - CSI does not account for correct negatives (true negatives)
    - For rare events, CSI can be low even for skillful forecasts

For ensemble forecasts, the ensemble mean is used as the deterministic
forecast before applying the threshold.

References:
    [1] Schaefer, J. T. (1990). The critical success index as an indicator
        of warning skill. Weather and Forecasting, 5(4), 570-575.
    [2] Wilks, D. S. (2011). Statistical Methods in the Atmospheric Sciences
        (3rd ed.). Academic Press.
"""

import os
import numpy as np
import h5py
from typing import Tuple, Optional, List, Union

DEFAULT_THRESHOLDS = [2, 4, 6, 8, 10]  # mm/hr


def csi(
    predictions: np.ndarray,
    targets: np.ndarray,
    thresholds: Optional[List[float]] = None,
    ensemble_axis: Optional[int] = None,
) -> dict:
    """
    Compute the Critical Success Index (CSI) at given intensity thresholds.

    Handles both deterministic predictions and ensemble predictions. For
    ensemble predictions, the ensemble mean is computed first to produce a
    single deterministic forecast, then CSI is calculated per threshold.

    Parameters
    ----------
    predictions : np.ndarray
        Forecast values. For deterministic forecasts the shape should match
        ``targets``. For ensemble forecasts, one extra dimension (the ensemble
        dimension) is present; specify which axis it sits on with
        ``ensemble_axis``.
    targets : np.ndarray
        Observed values. Shape must match ``predictions`` after the ensemble
        dimension (if any) is collapsed.
    thresholds : list of float, optional
        Intensity thresholds in the same units as the data (e.g. mm/hr).
        Defaults to [2, 4, 6, 8, 10].
    ensemble_axis : int, optional
        Axis of the ensemble dimension in ``predictions``. If None, the
        predictions are treated as deterministic (no ensemble dimension).

    Returns
    -------
    dict
        Dictionary keyed by threshold value. Each entry is a dict with:
        - 'csi'  : float – overall CSI score
        - 'tp'   : int   – total true positives
        - 'fp'   : int   – total false positives
        - 'fn'   : int   – total false negatives
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # Collapse ensemble dimension via the mean
    if ensemble_axis is not None:
        preds = np.mean(predictions, axis=ensemble_axis)
    else:
        preds = np.asarray(predictions)

    targets = np.asarray(targets)

    results = {}
    for thr in thresholds:
        pred_binary = (preds >= thr)
        obs_binary = (targets >= thr)

        tp = int(np.sum(pred_binary & obs_binary))
        fp = int(np.sum(pred_binary & ~obs_binary))
        fn = int(np.sum(~pred_binary & obs_binary))

        denominator = tp + fp + fn
        csi_value = tp / denominator if denominator > 0 else np.nan

        results[thr] = {
            'csi': float(csi_value),
            'tp': tp,
            'fp': fp,
            'fn': fn,
        }

    return results


def evaluate_csi_from_h5(
    input_file: str,
    output_dir: Optional[str] = None,
    thresholds: Optional[List[float]] = None,
    batch_size: int = 100,
) -> Tuple[dict, dict]:
    """
    Evaluate CSI from forecasts stored in an HDF5 file.

    Reads predictions and targets from an HDF5 file created by
    ``run_nowcast.py``, computes CSI at each threshold, and saves the results.

    Supports both ensemble and deterministic (mean) prediction files:
        - Ensemble mode: predictions (N, E, T, H, W), targets (N, T, H, W)
        - Deterministic mode:     predictions (N, T, H, W),    targets (N, T, H, W)

    Parameters
    ----------
    input_file : str
        Path to the HDF5 file containing 'predictions' and 'targets' datasets.
    output_dir : str, optional
        Directory to save results. If None, saves alongside ``input_file``.
    thresholds : list of float, optional
        Intensity thresholds (mm/hr). Defaults to [2, 4, 6, 8, 10].
    batch_size : int, default=100
        Number of samples to load at a time to manage memory.

    Returns
    -------
    csi_per_timestep : dict
        ``{threshold: np.ndarray}`` where each array has shape ``(T,)``
        giving the CSI at that threshold for each forecast lead time.
    summary : dict
        Summary statistics including per-threshold overall CSI, per-timestep
        CSI, contingency counts, and output file paths.

    Examples
    --------
    >>> results, summary = evaluate_csi_from_h5(
    ...     'nowcaster_results/forecast_results/steps_ensemble_20/test/results.h5'
    ... )
    >>> print(summary['overall_csi'])
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    os.makedirs(output_dir, exist_ok=True)

    # --- Read metadata ---------------------------------------------------
    with h5py.File(input_file, 'r') as f:
        ensemble_mode = f.attrs.get('ensemble_mode', False)
        pred_shape = f['predictions'].shape
        target_shape = f['targets'].shape

    if ensemble_mode:
        # predictions: (N, E, T, H, W)
        n_samples, n_ensemble, n_timesteps, height, width = pred_shape
    else:
        # predictions: (N, T, H, W)
        n_samples, n_timesteps, height, width = pred_shape
        n_ensemble = None

    print("=" * 60)
    print("CSI Evaluation")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Ensemble mode: {ensemble_mode}")
    if ensemble_mode:
        print(f"Number of ensemble members: {n_ensemble}")
    print(f"Predictions shape: {pred_shape}")
    print(f"Targets shape: {target_shape}")
    print(f"Number of samples: {n_samples}")
    print(f"Forecast timesteps: {n_timesteps}")
    print(f"Spatial dimensions: {height} x {width}")
    print(f"Thresholds (mm/hr): {thresholds}")
    print("=" * 60)

    # --- Accumulate contingency counts per (threshold, timestep) ---------
    # We accumulate TP, FP, FN globally and per-timestep so that the final
    # CSI is computed from pooled counts (micro-averaged), which is the
    # standard approach for spatial verification.
    tp_counts = {thr: np.zeros(n_timesteps, dtype=np.int64) for thr in thresholds}
    fp_counts = {thr: np.zeros(n_timesteps, dtype=np.int64) for thr in thresholds}
    fn_counts = {thr: np.zeros(n_timesteps, dtype=np.int64) for thr in thresholds}

    n_batches = (n_samples + batch_size - 1) // batch_size

    with h5py.File(input_file, 'r') as f:
        predictions = f['predictions']
        targets = f['targets']

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)

            pred_batch = predictions[start:end]   # (B, E, T, H, W) or (B, T, H, W)
            target_batch = targets[start:end]      # (B, T, H, W)

            # Collapse ensemble to mean if present
            if ensemble_mode:
                pred_batch = np.mean(pred_batch, axis=1)  # -> (B, T, H, W)

            # Iterate over timesteps
            for t in range(n_timesteps):
                pred_t = pred_batch[:, t, :, :]     # (B, H, W)
                target_t = target_batch[:, t, :, :]  # (B, H, W)

                for thr in thresholds:
                    p = pred_t >= thr
                    o = target_t >= thr
                    tp_counts[thr][t] += int(np.sum(p & o))
                    fp_counts[thr][t] += int(np.sum(p & ~o))
                    fn_counts[thr][t] += int(np.sum(~p & o))

            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                print(f"Processed {end}/{n_samples} samples...")

    # --- Compute CSI from pooled counts ----------------------------------
    csi_per_timestep = {}
    overall_csi = {}
    contingency = {}

    for thr in thresholds:
        tp = tp_counts[thr]
        fp = fp_counts[thr]
        fn = fn_counts[thr]
        denom = tp + fp + fn

        csi_t = np.where(denom > 0, tp / denom, np.nan)
        csi_per_timestep[thr] = csi_t

        tp_total = int(tp.sum())
        fp_total = int(fp.sum())
        fn_total = int(fn.sum())
        denom_total = tp_total + fp_total + fn_total
        overall = tp_total / denom_total if denom_total > 0 else np.nan

        overall_csi[thr] = float(overall)
        contingency[thr] = {
            'tp': tp_total,
            'fp': fp_total,
            'fn': fn_total,
            'tp_per_timestep': tp.tolist(),
            'fp_per_timestep': fp.tolist(),
            'fn_per_timestep': fn.tolist(),
        }

    # --- Save results ----------------------------------------------------
    csi_output_file = os.path.join(output_dir, 'csi_values.h5')
    with h5py.File(csi_output_file, 'w') as f:
        for thr in thresholds:
            grp = f.create_group(f"threshold_{thr}")
            grp.create_dataset('csi_per_timestep', data=csi_per_timestep[thr])
            grp.create_dataset('tp_per_timestep', data=tp_counts[thr])
            grp.create_dataset('fp_per_timestep', data=fp_counts[thr])
            grp.create_dataset('fn_per_timestep', data=fn_counts[thr])
            grp.attrs['overall_csi'] = overall_csi[thr]
        f.attrs['thresholds'] = thresholds
        f.attrs['n_samples'] = n_samples
        f.attrs['n_timesteps'] = n_timesteps
        f.attrs['ensemble_mode'] = ensemble_mode
        if n_ensemble is not None:
            f.attrs['n_ensemble'] = n_ensemble

    summary_output_file = os.path.join(output_dir, 'csi_summary.npz')
    save_dict = {
        'thresholds': np.array(thresholds),
        'n_samples': n_samples,
        'n_timesteps': n_timesteps,
    }
    for thr in thresholds:
        save_dict[f'csi_per_timestep_{thr}'] = csi_per_timestep[thr]
        save_dict[f'overall_csi_{thr}'] = overall_csi[thr]
    np.savez(summary_output_file, **save_dict)

    # --- Build summary dict ----------------------------------------------
    summary = {
        'overall_csi': overall_csi,
        'csi_per_timestep': {thr: csi_per_timestep[thr].tolist() for thr in thresholds},
        'contingency': contingency,
        'thresholds': thresholds,
        'n_samples': n_samples,
        'n_timesteps': n_timesteps,
        'ensemble_mode': ensemble_mode,
        'n_ensemble': n_ensemble,
        'csi_output_file': csi_output_file,
        'summary_output_file': summary_output_file,
    }

    # --- Print results ---------------------------------------------------
    print("\n" + "=" * 60)
    print("CSI Evaluation Results")
    print("=" * 60)
    for thr in thresholds:
        print(f"\nThreshold: {thr} mm/hr  |  Overall CSI: {overall_csi[thr]:.4f}")
        print(f"  TP={contingency[thr]['tp']}  FP={contingency[thr]['fp']}  FN={contingency[thr]['fn']}")
        print("  Per timestep:")
        for t, val in enumerate(csi_per_timestep[thr]):
            print(f"    t+{t+1}: {val:.4f}")
    print(f"\nResults saved to:")
    print(f"  - CSI values: {csi_output_file}")
    print(f"  - Summary:    {summary_output_file}")
    print("=" * 60)

    return csi_per_timestep, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate CSI from forecast HDF5 files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate CSI with default thresholds (2,4,6,8,10 mm/hr)
  python csi.py --input forecast_results/steps_ensemble_20/test/results.h5

  # Custom thresholds
  python csi.py --input results.h5 --thresholds 1 5 10 20
        """
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="Path to HDF5 file with predictions and targets"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="Directory to save results (default: same as input)"
    )
    parser.add_argument(
        '--thresholds',
        type=float,
        nargs='+',
        default=None,
        help="Thresholds in mm/hr (default: 2 4 6 8 10)"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help="Batch size for processing (default: 100)"
    )

    args = parser.parse_args()

    evaluate_csi_from_h5(
        args.input,
        output_dir=args.output_dir,
        thresholds=args.thresholds,
        batch_size=args.batch_size,
    )
