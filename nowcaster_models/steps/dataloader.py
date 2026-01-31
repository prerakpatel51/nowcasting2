"""PyTorch Lightning DataLoader for IMERG precipitation data."""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, Tuple

from .config import DataConfig


class PrecipitationDataset(Dataset):
    """Dataset for loading precipitation images from HDF5 file.

    Args:
        h5_path: Path to the HDF5 file containing precipitation data.
        indices: Array of indices to use for this dataset split.
        seq_length: Number of frames in each sequence (input + output).
        input_frames: Number of input frames.
        transform: Optional transform to apply to the data.
    """

    def __init__(
        self,
        h5_path: str,
        indices: np.ndarray,
        seq_length: int = 12,
        input_frames: int = 6,
        transform: Optional[callable] = None,
    ):
        self.h5_path = h5_path
        self.indices = indices
        self.seq_length = seq_length
        self.input_frames = input_frames
        self.output_frames = seq_length - input_frames
        self.transform = transform

        # Get total length from file
        with h5py.File(h5_path, 'r') as f:
            self.total_frames = f['precipitation'].shape[0]

        # Filter indices to ensure we can form complete sequences
        self.valid_indices = self._get_valid_sequence_indices()

    def _get_valid_sequence_indices(self) -> np.ndarray:
        """Filter indices to only include those that can form complete sequences."""
        valid = []
        for idx in self.indices:
            if idx + self.seq_length <= self.total_frames:
                valid.append(idx)
        return np.array(valid)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = self.valid_indices[idx]

        with h5py.File(self.h5_path, 'r') as f:
            sequence = f['precipitation'][start_idx:start_idx + self.seq_length]

        sequence = sequence.astype(np.float32)

        if self.transform is not None:
            sequence = self.transform(sequence)

        input_frames = sequence[:self.input_frames]
        target_frames = sequence[self.input_frames:]

        # Add channel dimension: (T, H, W) -> (T, 1, H, W)
        input_frames = torch.from_numpy(input_frames).unsqueeze(1)
        target_frames = torch.from_numpy(target_frames).unsqueeze(1)

        return input_frames, target_frames


class PrecipitationDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for precipitation data.

    Args:
        config: DataConfig object with all data loading parameters.
                If None, uses default DataConfig.
    """

    def __init__(self, config: Optional[DataConfig] = None):
        super().__init__()
        self.config = config or DataConfig()

        self.train_dataset: Optional[PrecipitationDataset] = None
        self.val_dataset: Optional[PrecipitationDataset] = None
        self.test_dataset: Optional[PrecipitationDataset] = None

    def setup(self, stage: Optional[str] = None):
        """Set up train, validation, and test datasets."""
        cfg = self.config

        # Get total number of samples
        with h5py.File(cfg.h5_path, 'r') as f:
            total_samples = f['precipitation'].shape[0]

        # Sequential split: train on past, validate/test on future
        # This prevents data leakage and simulates real-world deployment
        n_train_frames = int(total_samples * cfg.train_ratio)
        n_val_frames = int(total_samples * cfg.val_ratio)

        # Create indices with stride for each split
        train_indices = np.arange(0, n_train_frames - cfg.seq_length, cfg.stride)
        val_indices = np.arange(n_train_frames, n_train_frames + n_val_frames - cfg.seq_length, cfg.stride)
        test_indices = np.arange(n_train_frames + n_val_frames, total_samples - cfg.seq_length, cfg.stride)

        if stage == 'fit' or stage is None:
            self.train_dataset = PrecipitationDataset(
                h5_path=cfg.h5_path,
                indices=train_indices,
                seq_length=cfg.seq_length,
                input_frames=cfg.input_frames,
            )
            self.val_dataset = PrecipitationDataset(
                h5_path=cfg.h5_path,
                indices=val_indices,
                seq_length=cfg.seq_length,
                input_frames=cfg.input_frames,
            )

        if stage == 'test' or stage is None:
            self.test_dataset = PrecipitationDataset(
                h5_path=cfg.h5_path,
                indices=test_indices,
                seq_length=cfg.seq_length,
                input_frames=cfg.input_frames,
            )

    def train_dataloader(self) -> DataLoader:
        cfg = self.config
        return DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle_train,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        cfg = self.config
        return DataLoader(
            self.val_dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle_val,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        cfg = self.config
        return DataLoader(
            self.test_dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle_test,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        )


if __name__ == "__main__":
    # Example usage with default config
    from config import DataConfig

    config = DataConfig(num_workers=0)
    dm = PrecipitationDataModule(config=config)
    dm.setup()

    print(f"Train samples: {len(dm.train_dataset)}")
    print(f"Val samples: {len(dm.val_dataset)}")
    print(f"Test samples: {len(dm.test_dataset)}")

    train_loader = dm.train_dataloader()
    inputs, targets = next(iter(train_loader))
    print(f"Input shape: {inputs.shape}")   # (B, T_in, 1, H, W)
    print(f"Target shape: {targets.shape}") # (B, T_out, 1, H, W)
