"""Configuration for Lagrangian Persistence nowcasting."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """Data loading configuration parameters."""

    # Data paths
    h5_path: str = "/home1/ppatel2025/nowcasting2/data/imerg_data_h5_clean/imerg_data.h5"

    # Sequence settings
    seq_length: int = 24          # Total frames per sequence (input + output)
    input_frames: int = 12         # Number of input frames
    output_frames: int = 12        # Number of output/target frames
    stride: int = 12               # Step between consecutive sequences (1 = max overlap)

    # Shuffle settings (for time series: shuffle train, not val/test)
    shuffle_train: bool = True    # Shuffle training data
    shuffle_val: bool = False     # Don't shuffle validation (temporal evaluation)
    shuffle_test: bool = False    # Don't shuffle test (temporal evaluation)

    # Data split ratios
    train_ratio: float = 0.70     # 70% for training
    val_ratio: float = 0.15       # 15% for validation
    test_ratio: float = 0.15      # 15% for testing

    # DataLoader settings
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0, \
            "Split ratios must sum to 1.0"
        assert self.input_frames + self.output_frames == self.seq_length, \
            "input_frames + output_frames must equal seq_length"


@dataclass
class LPConfig:
    """Lagrangian Persistence algorithm configuration parameters."""

    # Precipitation thresholds
    precip_threshold: float = 0.1    # mm/h threshold for dB transform
    zerovalue: float = -15.0         # dB value for no-rain

    # Spatial/temporal resolution (for IMERG 64x64 over Burkina Faso)
    kmperpixel: float = 13.5         # km per pixel (8° / 64 pixels)
    timestep: int = 12               # Steps to predict


@dataclass
class Config:
    """Complete configuration combining all settings."""

    data: DataConfig = field(default_factory=DataConfig)
    lp: LPConfig = field(default_factory=LPConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create Config from a dictionary."""
        data_config = DataConfig(**config_dict.get("data", {}))
        lp_config = LPConfig(**config_dict.get("lp", {}))
        return cls(data=data_config, lp=lp_config)
