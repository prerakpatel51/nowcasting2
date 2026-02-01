# Lagrangian Persistence Nowcasting

Lagrangian Persistence is a deterministic advection-based precipitation nowcasting method that extrapolates the most recent observation along estimated motion vectors.

## What Lagrangian Persistence Does

1. **dB Transform**: Converts precipitation (mm/h) to decibels for better statistical properties
2. **Motion Estimation**: Uses Lucas-Kanade optical flow to estimate how rain is moving
3. **Extrapolation**: Projects the last observation forward in time using the motion field
4. **Back Transform**: Converts forecast back to mm/h

Unlike STEPS, Lagrangian Persistence is **deterministic** (no ensemble) and assumes the precipitation field remains constant along its trajectory (persistence assumption).

## Data Loading

### Quick Start

```python
from nowcaster_models.lagrangian_persistence import PrecipitationDataModule, DataConfig

# Option 1: Use defaults
dm = PrecipitationDataModule()
dm.setup()

# Option 2: Custom configuration
config = DataConfig(
    batch_size=16,
    input_frames=4,
    output_frames=8,
    seq_length=12,
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
| `seq_length` | 24 | Total frames per sequence |
| `input_frames` | 12 | Number of input frames |
| `output_frames` | 12 | Number of target frames |
| `stride` | 12 | Step between consecutive sequences |
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

## Lagrangian Persistence Algorithm Usage

```python
from nowcaster_models.lagrangian_persistence import lagrangian_persistence, LPConfig

# Load your IMERG data: shape (T, H, W) in mm/h
precip = load_imerg_data()  # e.g., shape (12, 64, 64)

# Option 1: Use defaults (configured for IMERG 64x64 Burkina Faso data)
forecast = lagrangian_persistence(precip, timesteps=12)

# Option 2: Custom configuration
config = LPConfig(
    precip_threshold=0.1,
    zerovalue=-15.0,
)
forecast = lagrangian_persistence(precip, timesteps=12, config=config)
```

## LP Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `precip_threshold` | 0.1 | mm/h threshold for rain detection |
| `zerovalue` | -15.0 | dB value for no-rain pixels |
| `kmperpixel` | 13.5 | Spatial resolution (km per pixel) |
| `timestep` | 12 | Number of time steps to predict |

### Default Values Explanation

The defaults are configured for IMERG data over Burkina Faso:

- **kmperpixel = 13.5**: Calculated from 8 longitude / 64 pixels x 108 km/deg
- **precip_threshold = 0.1**: Standard threshold to distinguish rain from no-rain

## Combined Configuration

Use the `Config` class to manage both data and LP settings:

```python
from nowcaster_models.lagrangian_persistence import Config, DataConfig, LPConfig

# Create combined config
config = Config(
    data=DataConfig(batch_size=64, num_workers=8),
    lp=LPConfig(precip_threshold=0.05),
)

# Access settings
print(config.data.batch_size)  # 64
print(config.lp.precip_threshold)  # 0.05
```

## Input Requirements

- **Shape**: `(T, H, W)` where T >= 2 frames (need at least 2 for motion estimation)
- **Units**: mm/h (precipitation rate)
- **Order**: Oldest frame first, newest last

## Output

- **Shape**: `(timesteps, H, W)`
- **Units**: mm/h
- **Value**: Deterministic extrapolation forecast (guaranteed no NaN values)

## Key Concepts

### Motion Estimation (Lucas-Kanade)
The Lucas-Kanade optical flow algorithm estimates the motion field by tracking features between consecutive frames. It assumes:
- Brightness constancy (precipitation intensity is preserved)
- Small motion between frames
- Spatial coherence (neighboring pixels move together)

### Extrapolation
The last observed precipitation field is advected forward using the estimated motion vectors. Each pixel is displaced according to the velocity field for each forecast time step.

### Persistence Assumption
Lagrangian persistence assumes that precipitation intensity remains constant along trajectories. This works well for:
- Short lead times (0-2 hours)
- Organized precipitation systems
- Steady-state conditions

The assumption breaks down for:
- Longer lead times (growth/decay not captured)
- Convective initiation/dissipation
- Complex terrain effects

## Comparison with STEPS

| Feature | Lagrangian Persistence | STEPS |
|---------|----------------------|-------|
| Output type | Deterministic | Probabilistic (ensemble) |
| Computation | Fast | Slower (multiple members) |
| Growth/decay | Not modeled | Stochastic perturbations |
| Best for | Short-term, baselines | Uncertainty quantification |

## References

- Germann & Zawadzki (2002): Scale-dependence of the predictability of precipitation
- pysteps documentation: https://pysteps.readthedocs.io/
