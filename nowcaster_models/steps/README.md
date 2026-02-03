# STEPS Nowcasting

STEPS (Short-Term Ensemble Prediction System) is a probabilistic precipitation nowcasting algorithm.

## What STEPS Does

1. **dB Transform**: Converts precipitation (mm/h) to decibels for better statistical properties
2. **Motion Estimation**: Uses Lucas-Kanade optical flow to estimate how rain is moving
3. **Ensemble Forecast**: Generates multiple possible futures with stochastic perturbations
4. **Back Transform**: Converts forecast back to mm/h

## Data Loading

### Quick Start

```python
from nowcaster_models.steps import PrecipitationDataModule, DataConfig

# Option 1: Use defaults
dm = PrecipitationDataModule()
dm.setup()

# Option 2: Custom configuration
config = DataConfig(
    batch_size=16,
    input_frames=12,
    output_frames=12,
    seq_length=24,
    stride=6,  # More overlap between sequences
)
dm = PrecipitationDataModule(config=config)
dm.setup()

# Use with PyTorch Lightning Trainer
trainer.fit(model, datamodule=dm)
```

### Data Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `h5_path` | `data/imerg_data_h5_clean/imerg_data.h5` | Path to HDF5 file |
| `seq_length` | 24 | Total frames per sequence (input + output) |
| `input_frames` | 12 | Number of input frames |
| `output_frames` | 12 | Number of target frames |
| `stride` | 12 | Step between consecutive sequences (1 = max overlap) |
| `shuffle_train` | True | Shuffle training data |
| `shuffle_val` | False | Don't shuffle validation (temporal evaluation) |
| `shuffle_test` | False | Don't shuffle test (temporal evaluation) |
| `train_ratio` | 0.70 | Training data fraction |
| `val_ratio` | 0.15 | Validation data fraction |
| `test_ratio` | 0.15 | Test data fraction |
| `batch_size` | 32 | Batch size |
| `num_workers` | 4 | DataLoader workers |
| `pin_memory` | True | Pin memory for GPU transfer |
| `persistent_workers` | True | Keep workers alive between epochs |
| `seed` | 42 | Random seed for reproducibility |

### Data Format

**HDF5 Structure:**
- `precipitation`: `(N, 64, 64)` - Precipitation images in mm/h
- `timestamps`: `(N,)` - Unix timestamps
- `datetime_strings`: `(N,)` - Human-readable timestamps

**Output Tensors:**
- Input: `(batch, input_frames, 1, 64, 64)`
- Target: `(batch, output_frames, 1, 64, 64)`

### Dataset Statistics

With default configuration (70/15/15 split):
- Training: ~144,000 samples
- Validation: ~31,000 samples
- Test: ~31,000 samples

## STEPS Algorithm Usage

```python
from nowcaster_models.steps import steps, STEPSConfig

# Load your IMERG data: shape (T, H, W) in mm/h
precip = load_imerg_data()  # e.g., shape (3, 64, 64)

# Option 1: Use defaults (configured for IMERG 64x64 Burkina Faso data)
forecast = steps(precip, timesteps=6)

# Option 2: Custom configuration
config = STEPSConfig(
    n_ensemble=30,
    n_cascade_levels=8,
    return_ensemble=False,  # Return mean instead of all members
)
forecast = steps(precip, timesteps=6, config=config)
```

## STEPS Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_ensemble` | 20 | Number of ensemble members to generate |
| `return_ensemble` | True | True: return all members, False: return mean |
| `n_cascade_levels` | 6 | Spatial scale decomposition levels |
| `precip_threshold` | 0.1 | mm/h threshold for dB transform |
| `precip_thr` | -10.0 | dB threshold for STEPS internal use |
| `zerovalue` | -15.0 | dB value for no-rain pixels |
| `kmperpixel` | 13.5 | Spatial resolution (km per pixel) |
| `timestep` | 12 | Number of frames to predict |

### Default Values Explanation

The defaults are configured for IMERG data over Burkina Faso:

- **kmperpixel = 13.5**: Calculated from 8 longitude / 64 pixels x 108 km/deg
- **timestep = 12**: Time between frames in minutes

## Combined Configuration

Use the `Config` class to manage both data and STEPS settings:

```python
from nowcaster_models.steps import Config, DataConfig, STEPSConfig

# Create combined config
config = Config(
    data=DataConfig(batch_size=64, num_workers=8, stride=6),
    steps=STEPSConfig(n_ensemble=30, return_ensemble=True),
)

# Access settings
print(config.data.batch_size)  # 64
print(config.steps.n_ensemble)  # 30
```

## Input Requirements

- **Shape**: `(T, H, W)` where T >= 3 frames
- **Units**: mm/h (precipitation rate)
- **Order**: Oldest frame first, newest last

## Output

- **Shape**: `(n_ensemble, timesteps, H, W)` if `return_ensemble=True`, else `(timesteps, H, W)`
- **Units**: mm/h
- **Value**: All ensemble members if `return_ensemble=True`, otherwise ensemble mean forecast

## Key Concepts

### Ensemble Members
STEPS generates N different possible forecasts (ensemble members). Each member has slightly different:
- Stochastic noise (random perturbations)
- Motion field perturbations

The final output is the **mean** across all members.

### Cascade Levels
The precipitation field is decomposed into multiple spatial scales:
- **Level 1**: Large-scale patterns (100s km) - more predictable
- **Level N**: Small-scale features (10s km) - decay faster

### dB Transform
Precipitation data is highly skewed (many zeros, few large values). The log transform:
```
dB = 10 * log10(precip)
```
makes the distribution more Gaussian, which works better for the stochastic model.

## References

- Bowler et al. (2006): STEPS algorithm paper
- pysteps documentation: https://pysteps.readthedocs.io/
