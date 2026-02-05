"""
Nowcasting Model Inference Script
=================================

This script runs inference for precipitation nowcasting models (STEPS and
Lagrangian Persistence) on specified dataset splits and saves predictions.

Features:
    - Batch-based saving: Results saved every N samples for crash recovery
    - Resume capability: Automatically resumes from last saved batch if interrupted
    - Memory efficient: Only keeps one batch in memory at a time
    - Flexible: Supports multiple models and dataset splits
    - Ensemble support: STEPS can return individual ensemble members or mean
      (configured via STEPSConfig.n_ensemble and STEPSConfig.return_ensemble)

Usage:
    # Run all models on test set (default)
    python run_nowcast.py

    # Run specific model on specific split
    python run_nowcast.py --model steps --split val

    # Run LP on test set
    python run_nowcast.py --model lp

    # To change STEPS ensemble settings, edit STEPSConfig in:
    # nowcaster_models/steps/config.py

Output:
    Results saved to: nowcaster_results/forecast_results/{model}/{split}/
    - results.h5: HDF5 file containing 'predictions' and 'targets' datasets
      - Determinitic mode: predictions (N, T, H, W), targets (N, T, H, W)
      - Ensemble mode: predictions (N, E, T, H, W), targets (N, T, H, W)
        where E = number of ensemble members
    - failed_samples.txt: Log of any failed samples (if any)
"""

import argparse
import os
import numpy as np
import h5py
from tqdm import tqdm

# Batch size for saving results (saves every N samples)
SAVE_BATCH_SIZE = 10


def save_batch_to_h5(output_file: str, predictions: np.ndarray, targets: np.ndarray,
                     pred_shape: tuple, target_shape: tuple) -> None:
    """
    Append a batch of predictions and targets to an HDF5 file.

    Creates the file and datasets if they don't exist, otherwise appends to existing data.
    Supports both ensemble mode (predictions have extra ensemble dimension) and mean mode.

    Parameters
    ----------
    output_file : str
        Path to the HDF5 output file.
    predictions : np.ndarray
        Batch of predictions.
        - Deterministic mode: (batch_size, T, H, W)
        - Ensemble mode: (batch_size, n_ensemble, T, H, W)
    targets : np.ndarray
        Batch of targets with shape (batch_size, T, H, W).
    pred_shape : tuple
        Shape of a single prediction sample for dataset creation.
        - Mean mode: (T, H, W)
        - Ensemble mode: (n_ensemble, T, H, W)
    target_shape : tuple
        Shape of a single target sample (T, H, W) for dataset creation.
    """
    with h5py.File(output_file, 'a') as f:
        if 'predictions' not in f:
            # Create resizable datasets
            f.create_dataset(
                'predictions',
                data=predictions,
                maxshape=(None,) + pred_shape,
                chunks=(1,) + pred_shape
            )
            f.create_dataset(
                'targets',
                data=targets,
                maxshape=(None,) + target_shape,
                chunks=(1,) + target_shape
            )
            # Store metadata about ensemble mode
            f.attrs['ensemble_mode'] = len(pred_shape) == 4  # True if (E, T, H, W)
            if len(pred_shape) == 4:
                f.attrs['n_ensemble'] = pred_shape[0]
        else:
            # Append to existing datasets
            current_size = f['predictions'].shape[0]
            new_size = current_size + predictions.shape[0]

            f['predictions'].resize(new_size, axis=0)
            f['predictions'][current_size:new_size] = predictions

            f['targets'].resize(new_size, axis=0)
            f['targets'][current_size:new_size] = targets


def get_resume_index(output_file: str) -> int:
    """
    Get the number of samples already processed from existing results file.

    Parameters
    ----------
    output_file : str
        Path to the HDF5 output file.

    Returns
    -------
    int
        Number of samples already saved (0 if file doesn't exist).
    """
    if os.path.exists(output_file):
        with h5py.File(output_file, 'r') as f:
            return f['predictions'].shape[0]
    return 0


def run_steps(split: str, output_dir: str) -> None:
    """
    Run STEPS model inference on the specified dataset split.

    STEPS (Short-Term Ensemble Prediction System) is a probabilistic nowcasting
    method that generates ensemble forecasts using stochastic perturbations.

    Ensemble settings are read from STEPSConfig (n_ensemble, return_ensemble).

    Parameters
    ----------
    split : str
        Dataset split to use: 'train', 'val', or 'test'.
    output_dir : str
        Base directory for saving results.
    """
    # Import model-specific modules
    from nowcaster_models.steps.config import DataConfig, STEPSConfig
    from nowcaster_models.steps.dataloader import PrecipitationDataModule
    from nowcaster_models.steps.steps import steps

    # Initialize configs
    steps_config = STEPSConfig()
    n_ensemble = steps_config.n_ensemble
    return_ensemble = steps_config.return_ensemble

    print("=" * 50)
    print("STEPS Evaluation")
    print(f"Dataset split: {split}")
    print(f"Ensemble members: {n_ensemble}")
    print(f"Return ensemble: {return_ensemble}")
    print("=" * 50)

    # Initialize data module
    data_config = DataConfig(batch_size=1, num_workers=0)
    dm = PrecipitationDataModule(config=data_config)
    dm.setup(stage='fit' if split in ['train', 'val'] else 'test')

    # Select appropriate dataloader
    if split == 'train':
        dataloader = dm.train_dataloader()
    elif split == 'val':
        dataloader = dm.val_dataloader()
    else:
        dataloader = dm.test_dataloader()

    # Setup output paths - include ensemble info in directory name for ensemble mode
    if return_ensemble:
        save_dir = os.path.join(output_dir, f'steps_ensemble_{n_ensemble}', split)
    else:
        save_dir = os.path.join(output_dir, 'steps', split)
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, 'results.h5')
    failed_path = os.path.join(save_dir, 'failed_samples.txt')

    # Check for resume point
    start_idx = get_resume_index(output_file)
    if start_idx > 0:
        print(f"Resuming from sample {start_idx}")

    # Define shapes based on mode
    target_shape = (data_config.output_frames, 64, 64)
    if return_ensemble:
        pred_shape = (n_ensemble, data_config.output_frames, 64, 64)
    else:
        pred_shape = (data_config.output_frames, 64, 64)

    batch_predictions = []
    batch_targets = []
    processed_count = start_idx

    # Process samples
    for idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Running STEPS")):
        # Skip already processed samples
        if idx < start_idx:
            continue

        # Extract arrays: (B, T, 1, H, W) -> (T, H, W)
        input_seq = inputs[0].squeeze(1).numpy()
        input_seq = np.nan_to_num(input_seq, nan=0.0)
        target_seq = targets[0].squeeze(1).numpy()
        target_seq = np.nan_to_num(target_seq, nan=0.0)

        try:
            # Generate forecast (ensemble settings read from config)
            forecast = steps(
                input_seq,
                timesteps=data_config.output_frames,
                config=steps_config
            )

            # Add to batch buffer
            batch_predictions.append(forecast)
            batch_targets.append(target_seq)
            processed_count += 1

            # Save batch when buffer is full
            if len(batch_predictions) >= SAVE_BATCH_SIZE:
                save_batch_to_h5(
                    output_file,
                    np.stack(batch_predictions, axis=0),
                    np.stack(batch_targets, axis=0),
                    pred_shape,
                    target_shape
                )
                batch_predictions = []
                batch_targets = []

        except Exception as e:
            # Log failed sample
            with open(failed_path, 'a') as f:
                f.write(f"Sample {idx}: {e}\n")
            continue

    # Save remaining samples in buffer
    if batch_predictions:
        save_batch_to_h5(
            output_file,
            np.stack(batch_predictions, axis=0),
            np.stack(batch_targets, axis=0),
            pred_shape,
            target_shape
        )

    # Print summary
    with h5py.File(output_file, 'r') as f:
        print(f"\nResults saved to: {output_file}")
        print(f"Predictions shape: {f['predictions'].shape}")
        print(f"Targets shape: {f['targets'].shape}")
        if return_ensemble:
            print(f"Note: Each target has {n_ensemble} ensemble predictions")


def run_lp(split: str, output_dir: str) -> None:
    """
    Run Lagrangian Persistence model inference on the specified dataset split.

    Lagrangian Persistence extrapolates the last observed precipitation field
    along estimated motion vectors (advection-based persistence).

    Parameters
    ----------
    split : str
        Dataset split to use: 'train', 'val', or 'test'.
    output_dir : str
        Base directory for saving results.
    """
    # Import model-specific modules
    from nowcaster_models.lagrangian_persistence.config import DataConfig, LPConfig
    from nowcaster_models.lagrangian_persistence.dataloader import PrecipitationDataModule
    from nowcaster_models.lagrangian_persistence.lp import lagrangian_persistence

    print("=" * 50)
    print("Lagrangian Persistence Evaluation")
    print(f"Dataset split: {split}")
    print("=" * 50)

    # Initialize data module
    data_config = DataConfig(batch_size=1, num_workers=0)
    dm = PrecipitationDataModule(config=data_config)
    dm.setup(stage='fit' if split in ['train', 'val'] else 'test')

    # Select appropriate dataloader
    if split == 'train':
        dataloader = dm.train_dataloader()
    elif split == 'val':
        dataloader = dm.val_dataloader()
    else:
        dataloader = dm.test_dataloader()

    # Setup output paths
    save_dir = os.path.join(output_dir, 'lagrangian_persistence', split)
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, 'results.h5')
    failed_path = os.path.join(save_dir, 'failed_samples.txt')

    # Check for resume point
    start_idx = get_resume_index(output_file)
    if start_idx > 0:
        print(f"Resuming from sample {start_idx}")

    # Initialize model config and batch buffers
    lp_config = LPConfig()
    sample_shape = (data_config.output_frames, 64, 64)

    batch_predictions = []
    batch_targets = []
    processed_count = start_idx

    # Process samples
    for idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Running LP")):
        # Skip already processed samples
        if idx < start_idx:
            continue

        # Extract arrays: (B, T, 1, H, W) -> (T, H, W)
        input_seq = inputs[0].squeeze(1).numpy()
        input_seq = np.nan_to_num(input_seq, nan=0.0)
        target_seq = targets[0].squeeze(1).numpy()
        target_seq = np.nan_to_num(target_seq, nan=0.0)

        try:
            # Generate forecast
            forecast = lagrangian_persistence(input_seq, timesteps=data_config.output_frames, config=lp_config)

            # Add to batch buffer
            batch_predictions.append(forecast)
            batch_targets.append(target_seq)
            processed_count += 1

            # Save batch when buffer is full
            if len(batch_predictions) >= SAVE_BATCH_SIZE:
                save_batch_to_h5(
                    output_file,
                    np.stack(batch_predictions, axis=0),
                    np.stack(batch_targets, axis=0),
                    sample_shape,
                    sample_shape
                )
                batch_predictions = []
                batch_targets = []

        except Exception as e:
            # Log failed sample
            with open(failed_path, 'a') as f:
                f.write(f"Sample {idx}: {e}\n")
            continue

    # Save remaining samples in buffer
    if batch_predictions:
        save_batch_to_h5(
            output_file,
            np.stack(batch_predictions, axis=0),
            np.stack(batch_targets, axis=0),
            sample_shape,
            sample_shape
        )

    # Print summary
    with h5py.File(output_file, 'r') as f:
        print(f"\nResults saved to: {output_file}")
        print(f"Predictions shape: {f['predictions'].shape}")
        print(f"Targets shape: {f['targets'].shape}")


def main():
    """Parse command line arguments and run model inference."""
    # Default output directory relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(script_dir, 'forecast_results')

    parser = argparse.ArgumentParser(
        description="Run nowcasting model inference and save predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_nowcast.py                          # Run all models on test set
  python run_nowcast.py --model steps            # Run only STEPS on test set
  python run_nowcast.py --model lp --split val   # Run LP on validation set

  # STEPS ensemble settings are configured in:
  # nowcaster_models/steps/config.py (STEPSConfig.n_ensemble, STEPSConfig.return_ensemble)
        """
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        choices=['steps', 'lp', 'all'],
        help="Model to evaluate: 'steps', 'lp', or 'all' (default: all)"
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help="Dataset split to use: 'train', 'val', or 'test' (default: test)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=default_output,
        help="Output directory for saving results"
    )

    args = parser.parse_args()

    # Run selected model(s)
    if args.model == 'steps':
        run_steps(args.split, args.output_dir)
    elif args.model == 'lp':
        run_lp(args.split, args.output_dir)
    elif args.model == 'all':
        run_steps(args.split, args.output_dir)
        print("\n")
        run_lp(args.split, args.output_dir)


if __name__ == "__main__":
    main()
