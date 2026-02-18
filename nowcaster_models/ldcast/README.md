# LDCast IMERG Data Pipeline

This module adapts the [LDCast](https://github.com/MeteoSwiss/ldcast) training data pipeline to work with NASA IMERG precipitation data. It handles everything from raw HDF5 data to ready-to-use PyTorch Lightning DataLoaders for model training.

---

## Directory Structure

```
nowcaster_models/ldcast/
├── config.yaml          # Central configuration for all pipeline parameters
├── convert_imerg.py     # Step 1: HDF5 → NetCDF patch conversion
├── patches.py           # Patch I/O (save/load NetCDF files)
├── split.py             # Train/valid/test splitting + DataModule
├── transform.py         # Data normalization transforms
├── sampling.py          # Stratified (equal-frequency) sampler
├── batch.py             # Batch generation + PyTorch Dataset wrappers
├── dataloader.py        # Top-level setup_data() entry point
├── __init__.py          # Exports setup_data
├── data/IMERG/          # Patch NetCDF files (patches_IMERG_<year>.nc)
└── cache/               # Cached sampler pickles (sampler_{train,valid,test}.pkl)
```

---

## Configuration (`config.yaml`)

All parameters are controlled from a single YAML file:

```yaml
data:
  h5_path:   /path/to/imerg_data.h5   # Source HDF5 file
  patch_dir: data/IMERG/              # Where patch .nc files are stored
  cache_dir: cache/                   # Where sampler .pkl caches are stored
  var_name:  IMERG

patches:
  patch_size: 32        # Pixels per patch side
  image_size: 64        # Full frame side (64x64 → 2x2 grid of patches)
  stride: 32            # Patch stride (non-overlapping)
  min_nonzeros: 5       # Min non-zero pixels to classify as a "data" patch
  nonzero_threshold: 0.1

pipeline:
  input_timesteps: 16   # Number of past frames fed to the model (IMERG-O)
  output_timesteps: 12  # Number of future frames to predict  (IMERG-T)
  interval_minutes: 30  # Temporal resolution of IMERG
  sample_shape: [2, 2]  # Spatial grid of patches per sample

split:
  chunk_days: 2         # Temporal chunk size for train/valid/test splitting
  valid_frac: 0.1       # 10% for validation
  test_frac: 0.1        # 10% for test
  random_seed: 42

sampling:
  bins_low: 0.1         # Lower bound of intensity bins (mm/hr)
  bins_high: 20.0       # Upper bound of intensity bins (mm/hr)
  num_bins: 10          # Number of log-spaced bins
  batch_size: 32

transform:
  log: true             # Apply log10 before normalization
  threshold: 0.1        # Values below this are replaced with fill_value
  fill_value: 0.01      # Replacement value for sub-threshold pixels
  mean: -0.1399         # Dataset mean (in log10 space)
  std: 0.4151           # Dataset std  (in log10 space)
```

---

## Pipeline Overview (End-to-End Flow)

```
IMERG HDF5 file
      │
      ▼  convert_imerg.py
Yearly NetCDF patch files  (data/IMERG/patches_IMERG_<year>.nc)
      │
      ▼  patches.load_all_patches()
In-memory patch dictionary  {patches, patch_coords, patch_times, zero_*}
      │
      ▼  split.train_valid_test_split()
Per-split dictionaries  {train: {...}, valid: {...}, test: {...}}
      │
      ▼  transform.normalize_threshold()
Transform function  (threshold → log10 → standardize)
      │
      ▼  split.DataModule  (wraps batch.BatchGenerator)
PyTorch Lightning DataModule
      │
      ├── train_dataloader()  → StreamBatchDataset      (random, infinite)
      ├── val_dataloader()    → DeterministicBatchDataset (fixed seed)
      └── test_dataloader()   → DeterministicBatchDataset (fixed seed)
```

---

## Step 1 — Convert IMERG HDF5 to Patch NetCDF Files

Run once before training. Reads the consolidated HDF5 file and writes one NetCDF file per year into `data/IMERG/`.

```bash
cd nowcaster_models/ldcast
python convert_imerg.py                          # uses config.yaml by default
python convert_imerg.py --config /path/to/config.yaml
```

### How a frame is cut into patches

Each IMERG frame is 64×64 pixels. The converter slices it into a **2×2 grid** of non-overlapping 32×32 patches:

```
  Full 64×64 IMERG frame
  ┌─────────────┬─────────────┐
  │             │             │
  │  patch(0,0) │  patch(0,1) │   row i=0
  │   32×32     │   32×32     │
  ├─────────────┼─────────────┤
  │             │             │
  │  patch(1,0) │  patch(1,1) │   row i=1
  │   32×32     │   32×32     │
  └─────────────┴─────────────┘
     col j=0       col j=1
```

Each patch is independently classified:

```
n_nonzero = count pixels where patch > 0.1 mm/hr

if n_nonzero >= 5:
    → DATA patch   → store pixel array + (i, j) + timestamp
else:
    → ZERO patch   → store only (i, j) + timestamp  (no pixel data)
```

**Why separate zero patches?**
IMERG is dominated by dry frames. Storing only coordinates (not pixel data) for zero patches avoids huge redundant arrays of zeros while still allowing the sampler to know which positions are dry at any given time.

**Output format per NetCDF file:**

| Variable            | Shape                  | Description                      |
|---------------------|------------------------|----------------------------------|
| `patches`           | `(N, 32, 32)` float32  | Pixel data for non-zero patches  |
| `patch_coords`      | `(N, 2)` uint16        | `(row, col)` patch grid index    |
| `patch_times`       | `(N,)` int64           | Unix timestamps (seconds)        |
| `zero_patch_coords` | `(M, 2)` uint16        | Coords of dry patches            |
| `zero_patch_times`  | `(M,)` int64           | Timestamps of dry patches        |

---

## Step 2 — Set Up the DataModule for Training

```python
from omegaconf import OmegaConf
from nowcaster_models.ldcast import setup_data

config = OmegaConf.load("nowcaster_models/ldcast/config.yaml")
datamodule = setup_data(config)
```

### 2a. `patches.load_all_patches()` — Load all yearly NetCDF files

Scans `data/IMERG/` for `patches_IMERG_*.nc` files and uses **Dask multiprocessing** to load them in parallel, then concatenates into one big in-memory dictionary.

```python
raw = {
    "IMERG": {
        "patches":           np.ndarray,  # (N_total, 32, 32)  float32
        "patch_coords":      np.ndarray,  # (N_total, 2)       uint16
        "patch_times":       np.ndarray,  # (N_total,)         int64  Unix seconds
        "zero_patch_coords": np.ndarray,  # (M_total, 2)       uint16
        "zero_patch_times":  np.ndarray,  # (M_total,)         int64
    }
}
```

### 2b. `split.train_valid_test_split()` — Temporal chunk splitting

The full time range is divided into **2-day chunks**. Chunks are randomly shuffled (seed=42) and assigned:

```
Full timeline (2011 – 2022):

  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │chunk0│chunk1│chunk2│chunk3│chunk4│chunk5│chunk6│chunk7│ ...  │
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

  After random shuffle and assignment (80/10/10 split):
  ┌─────────┬────────┬────────┐
  │  train  │  valid │  test  │
  │  ~80%   │  ~10%  │  ~10%  │
  └─────────┴────────┴────────┘

  Note: chunks are NOT contiguous in time — they are randomly
  interleaved across the full 12-year span to prevent leakage.
```

Patches whose timestamps fall within a split's chunks are extracted into that split's dictionary.

### 2c. `transform.normalize_threshold()` — Build the transform function

Returns a **closure** (not yet applied — called later per batch). For each pixel value `x`:

```
Step 1 — Threshold clamp:
  if x < 0.1 mm/hr  →  x = 0.01

Step 2 — Log10:
  x = log10(x)

Step 3 — Standardize:
  x = (x - (-0.1399)) / 0.4151

Example values:
  0.0  mm/hr  →  clamp to 0.01  →  log10(0.01) = -2.0  →  (-2.0 + 0.1399)/0.4151 = -4.48
  0.1  mm/hr  →  log10(0.1)  = -1.0  →  (-1.0 + 0.1399)/0.4151 = -2.07
  1.0  mm/hr  →  log10(1.0)  =  0.0  →  ( 0.0 + 0.1399)/0.4151 =  0.34
  10.0 mm/hr  →  log10(10)   =  1.0  →  ( 1.0 + 0.1399)/0.4151 =  2.75
```

The inner kernel (`normalize_threshold_array`) is compiled by **Numba** and runs in parallel across CPU cores for speed.

### 2d. Variable definitions

Two named variables describe what the model sees and predicts:

| Name       | Role        | Timestep indices            | Frames | Time span        |
|------------|-------------|----------------------------|--------|------------------|
| `IMERG-O`  | Input (observation) | `[-15, -14, ..., 0]` | 16     | T−7.5h → T      |
| `IMERG-T`  | Target (prediction) | `[1, 2, ..., 12]`    | 12     | T+30min → T+6h  |

### 2e. `sampling.EqualFrequencySampler` — Intensity-stratified sampler

IMERG is dominated by dry/light-rain events. Without stratification the model would see mostly zeros and fail to learn heavy rain. The sampler fixes this by ensuring equal representation across intensity levels.

#### How bins are constructed

10 log-spaced bins between 0.1 and 20 mm/hr:

```
  Bin 0   Bin 1   Bin 2   Bin 3  ...  Bin 9   (overflow)
  ─────   ─────   ─────   ─────       ─────
  0.0     0.1     0.22    0.46  ...   9.3     20.0 mm/hr
     ↑         ↑       ↑
  dry/zero  drizzle  light        heavy rain →
```

Each data patch is classified by its **99th-percentile pixel value** into one of these bins. Zero patches go into bin 0.

#### How a sample is drawn

```
1. Pick a random bin (uniform across 10 bins)
2. Pick a random (t, i, j) starting position from that bin's list
3. Repeat for each item in the batch → equal frequency per bin
```

This means a heavy-rain patch (rare) is sampled just as often as a dry patch (common).

#### What is a "valid starting position"?

A `(t0, i0, j0)` is only stored as a candidate if **all 112 patch lookups** it requires are present in the index:

```
For ts in range(-16, 13):          # 29 timesteps (covers 16 input + 12 target + 1 buffer)
    for di in range(2):            # 2 patch rows
        for dj in range(2):        # 2 patch cols
            MUST have (t0 + ts*1800, i0+di, j0+dj) in patch_index
```

The sampler is cached to `cache/sampler_train.pkl` after the first build (which can take minutes) and reloaded on subsequent runs.

### 2f. `split.DataModule` — PyTorch Lightning DataModule

| Split | Dataset type              | Sampling         | Augmentation |
|-------|---------------------------|------------------|--------------|
| train | `StreamBatchDataset`      | Random per epoch | Yes (flip, transpose) |
| valid | `DeterministicBatchDataset` | Fixed seed 1234 | No          |
| test  | `DeterministicBatchDataset` | Fixed seed 2345 | No          |

---

## How the Spatial + Temporal Window is Built

This is the core of the pipeline. Every sample is defined by a single anchor `(t0, i0, j0)`.

### Spatial window (always the full frame in this config)

The `sample_shape=[2,2]` box is placed with its top-left corner at `(i0, j0)`:

```
  Patch grid  (2×2 for a 64×64 frame)

  ┌─────────────┬─────────────┐
  │ (i0,  j0)  │ (i0,  j0+1)│
  │  patch[0,0] │  patch[0,1] │   ← assembled into top half of output frame
  ├─────────────┼─────────────┤
  │ (i0+1,j0)  │ (i0+1,j0+1)│
  │  patch[1,0] │  patch[1,1] │   ← assembled into bottom half of output frame
  └─────────────┴─────────────┘

  Output pixel layout (64×64):
  ┌────────────────────────────────────┐
  │  patch(i0,j0)  │ patch(i0,j0+1)  │  pixels [0:32,  0:32 ] | [0:32,  32:64]
  ├────────────────────────────────────┤
  │ patch(i0+1,j0) │patch(i0+1,j0+1) │  pixels [32:64, 0:32 ] | [32:64, 32:64]
  └────────────────────────────────────┘
```

Since `image_size=64` and `sample_shape=[2,2]`, `i0` and `j0` can only ever be `0`. Every sample covers the **entire 64×64 frame**. Spatial diversity comes only from **which timestep** `t0` is chosen.

### Temporal window — the full 28-frame sequence

Every sample is anchored at a single moment `t0` (an IMERG timestamp). From there, 16 frames are taken **backwards** (the model's input) and 12 frames are taken **forwards** (the prediction target). The step between frames is always 30 minutes.

```
  Concrete example: anchor t0 = 12:00 UTC

  Wall-clock times (every tick = 30 min):

  04:30  05:00  05:30  ...  10:30  11:00  11:30  12:00  12:30  13:00  ...  17:30  18:00
    │      │      │           │      │      │      │      │      │           │      │
    ▼      ▼      ▼           ▼      ▼      ▼      ▼      ▼      ▼           ▼      ▼
  [ -15 ][-14 ][-13 ]  ... [ -3 ] [ -2 ] [ -1 ] [  0 ] [ +1 ] [ +2 ]  ... [+11 ] [+12]
    ●      ●      ●           ●      ●      ●      ●      ●      ●           ●      ●
    │                                             ↑│                               │
    │                                       anchor t0                              │
    │◄──────────────── IMERG-O (observation) ─────┤◄── IMERG-T (prediction) ──────┘
                       16 input frames                   12 target frames
                      T−7.5h  →  T                      T+30min  →  T+6h

  Each ● is one fully assembled 64×64 frame (stitched from 4 patches).
  The vertical bar at t0 is NOT a frame — it is just the boundary
  between past (input) and future (target).
```

#### Frame index to wall-clock mapping (example anchor = 12:00 UTC)

| Frame index | Offset | Wall clock | Role     |
|:-----------:|:------:|:----------:|:--------:|
| 0           | −15    | 04:30      | Input    |
| 1           | −14    | 05:00      | Input    |
| 2           | −13    | 05:30      | Input    |
| …           | …      | …          | Input    |
| 14          | −1     | 11:30      | Input    |
| 15          | 0      | **12:00**  | Input (most recent) |
| 16          | +1     | 12:30      | **Target** (first) |
| …           | …      | …          | Target   |
| 27          | +12    | 18:00      | **Target** (last)  |

Each `●` is one 64×64 frame assembled from 4 patches.

### Full window for one sample (e.g. sample 45)

```
  Anchor: (t0=T, i0=0, j0=0)

  IMERG-O  →  16 frames (input to the model):
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Frame  0: patches at (T-15×1800s, i=0,1, j=0,1)  →  64×64 pixels  │
  │ Frame  1: patches at (T-14×1800s, i=0,1, j=0,1)  →  64×64 pixels  │
  │   ...                                                               │
  │ Frame 15: patches at (T- 0×1800s, i=0,1, j=0,1)  →  64×64 pixels  │
  └─────────────────────────────────────────────────────────────────────┘
  Output shape: (batch=1, channels=1, timesteps=16, H=64, W=64)

  IMERG-T  →  12 frames (what the model must predict):
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Frame  0: patches at (T+ 1×1800s, i=0,1, j=0,1)  →  64×64 pixels  │
  │ Frame  1: patches at (T+ 2×1800s, i=0,1, j=0,1)  →  64×64 pixels  │
  │   ...                                                               │
  │ Frame 11: patches at (T+12×1800s, i=0,1, j=0,1)  →  64×64 pixels  │
  └─────────────────────────────────────────────────────────────────────┘
  Output shape: (batch=1, channels=1, timesteps=12, H=64, W=64)

  Total patch lookups: 4 spatial × 28 temporal = 112 per sample
```

### How `PatchIndex` assembles one frame (`build_batch` Numba kernel)

```python
# For each sample k in batch, each timestep bt:
for i in range(i0, i0+2):         # i = 0, 1
    for j in range(j0, j0+2):     # j = 0, 1
        ind = patch_index[(t, i, j)]
        if ind >= 0:               # real data patch
            out[k, bt, i*32:(i+1)*32, j*32:(j+1)*32] = patch_data[ind]
        elif ind == IDX_ZERO:      # dry patch — fill with zero_value
            out[k, bt, i*32:(i+1)*32, j*32:(j+1)*32] = 0
        elif ind == IDX_MISSING:   # patch not in index — fill with missing_value
            out[k, bt, i*32:(i+1)*32, j*32:(j+1)*32] = 0
```

The lookup dict maps `(t, i, j) → array index` and is built once at startup by two Numba JIT functions (`init_patch_index`, `init_patch_index_zero`).

---

## Step 3 — Using the DataLoaders

```python
# With PyTorch Lightning Trainer
trainer = pl.Trainer(...)
trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)

# Or iterate manually
for batch in datamodule.train_dataloader():
    predictors, target = batch
    # predictors: list containing one (data, t_relative) tuple
    #   data:       np.ndarray  (B, 1, 16, 64, 64)  float32  normalized
    #   t_relative: np.ndarray  (B, 16)              float32  time offsets
    # target:       np.ndarray  (B, 1, 12, 64, 64)  float32  normalized

for batch in datamodule.val_dataloader():   # deterministic, seed=1234
    ...

for batch in datamodule.test_dataloader():  # deterministic, seed=2345
    ...
```

### Batch tensor shapes at a glance

Each call to `batch_gen.batch()` returns a tuple `(pred_batch, target_batch)`:

```
  pred_batch  — list with one element per predictor variable (here, just IMERG-O):
  │
  └─ pred_batch[0]  =  (data, t_relative)
       │
       ├─ data        shape  (B, 1, 16, 64, 64)   float32   normalized pixel values
       │              │  B  = 32  (batch size)
       │              │  1  = number of channels (single-variable)
       │              │  16 = input timesteps  (frames -15 … 0)
       │              │  64 = height in pixels
       │              │  64 = width  in pixels
       │
       └─ t_relative  shape  (B, 16)               float32   time offsets (see below)

  target_batch  shape  (B, 1, 12, 64, 64)   float32   normalized target frames
                │  12 = output timesteps (frames +1 … +12)
                │  (no t_relative for the target — the model predicts the frames)
```

---

### What is `t_relative`?

`t_relative` is a **time-position encoding** that tells the model exactly how far each input frame is from the anchor moment `t0`, measured in units of `interval_secs` (one IMERG step = 1800 seconds = 30 min).

#### How it is computed (from `batch.py`)

```python
# t[s, ts] = absolute Unix timestamp of frame ts for sample s
t = t0_shifted[:, None] + interval_secs * timesteps[None, :]
#                                          ^^^^^^^^^^^^^^^^^^
#                         timesteps = [-15, -14, ..., -1, 0]  for IMERG-O

# Subtract each sample's own anchor and divide by the step size
t_relative = (t - t0[:, None]) / interval_secs
```

For IMERG-O (where `timesteps = np.arange(-15, 1)` and every anchor `t0` is already on the 30-min grid), `t_shift = 0` and the formula simplifies to:

```
t_relative[s, ts] = timesteps[ts]   for every sample s
```

So the 16 values in `t_relative[s]` are always the same integers, regardless of which `t0` was chosen:

```
  t_relative[s]  =  [-15., -14., -13., -12., -11., -10.,
                      -9.,  -8.,  -7.,  -6.,  -5.,  -4.,
                      -3.,  -2.,  -1.,   0.]

  Unit: one IMERG interval = 30 minutes
  So  -15 means "15 × 30 min = 7.5 hours before the anchor"
       -1 means "30 minutes before the anchor"
        0 means "at the anchor" (most recent observed frame)
```

#### What `t_relative` looks like in the batch

```
  t_relative  shape (B=32, T=16)

  Sample 0:  [-15. -14. -13. -12. -11. -10.  -9.  -8.  -7.  -6.  -5.  -4.  -3.  -2.  -1.   0.]
  Sample 1:  [-15. -14. -13. -12. -11. -10.  -9.  -8.  -7.  -6.  -5.  -4.  -3.  -2.  -1.   0.]
  ...        (identical for every sample in IMERG because all samples share the same
              input timestep structure — only t0 changes, not the relative offsets)
  Sample 31: [-15. -14. -13. -12. -11. -10.  -9.  -8.  -7.  -6.  -5.  -4.  -3.  -2.  -1.   0.]
```

#### Where does `t_relative` go?

`t_relative` is passed alongside the pixel data to the model. Attention-based models (like LDCast's U-Net with temporal attention) use it as a **positional embedding** so the model knows the temporal distance between frames — not just their order. This matters because:

- Without `t_relative`, the model only knows "frame 0, frame 1, …, frame 15" (ordinal positions).
- With `t_relative`, the model knows the actual time gap (30 min per step), enabling it to generalize across different temporal resolutions or missing frames.

```
  Model input flow:

  data        (B, 1, 16, 64, 64)  ──►  spatial encoder
                                         │
  t_relative  (B, 16)            ──►  time embedding layer
                                         │
                                   fused representation
                                         │
                                   temporal attention / U-Net
                                         │
                              prediction (B, 1, 12, 64, 64)
```

#### When would `t_relative` differ across samples?

In this IMERG-only setup it is always the same per-sample vector. It would differ if:
- A variable had a **coarser time step** (e.g. NWP at 6h instead of 30min): `t_shift` would align `t0` to the NWP grid, shifting the relative offsets.
- Frames were **missing** in the middle and the pipeline used irregular sampling.

### Data augmentation (training only)

Applied identically to all timesteps in a sample — the random draw happens once per batch call:

```
  augmentations() → (transpose, flipud, fliplr)  each 0 or 1

  Original:        Transpose:       FlipUD:         FlipLR:
  ┌───┬───┐        ┌───┬───┐        ┌───┬───┐        ┌───┬───┐
  │ A │ B │   →    │ A │ C │   →    │ C │ D │   →    │ B │ A │
  ├───┼───┤        ├───┼───┤        ├───┼───┤        ├───┼───┤
  │ C │ D │        │ B │ D │        │ A │ B │        │ D │ C │
  └───┴───┘        └───┴───┘        └───┴───┘        └───┴───┘
```

---

## Main Functions Reference

### `setup_data(config)` — `dataloader.py`

Top-level entry point. Orchestrates the full pipeline from patch loading to DataModule creation. Returns a `split.DataModule` ready for training.

**Call order:**
```
setup_data(config)
 ├─ patches.load_all_patches()        load + concatenate all yearly .nc files
 ├─ split.train_valid_test_split()    chunk-based temporal split
 ├─ transform.normalize_threshold()   build transform closure
 └─ split.DataModule(...)             wire everything into dataloaders
```

---

### `patches.load_all_patches(patch_dir, var)` — `patches.py`

Scans the patch directory for files matching `patches_{var}_*.nc`, loads them in parallel with Dask, and concatenates across years. All data lands in memory as NumPy arrays.

**Key behavior:** Uses `dask.delayed` + `scheduler="processes"` so each file is decompressed in a separate process, avoiding the GIL during NetCDF4 I/O.

---

### `split.get_chunks(primary_raw, ...)` — `split.py`

Divides the full time axis into fixed-size windows (2 days each) and randomly assigns them to train/valid/test. The random assignment means chunks from the same calendar period can land in different splits, preventing temporal leakage from nearby frames.

---

### `split.train_valid_test_split(raw_data, primary_raw_var, ...)` — `split.py`

Calls `get_chunks` to get split boundaries, then uses binary search (`bisect_left`) to efficiently extract patches whose timestamps fall in each split. Returns a nested dict `{split: {var: patch_dict}}`.

---

### `transform.normalize_threshold(...)` — `transform.py`

Returns a **stateful closure** that pre-allocates a float32 output buffer on first call and reuses it. The actual computation is done by `normalize_threshold_array`, a Numba `@njit(parallel=True)` kernel that processes elements in parallel across CPU cores.

---

### `sampling.EqualFrequencySampler(bins, patch_data, ...)` — `sampling.py`

The most expensive object to construct. Internally:

1. `bin_classify_patches_parallel()` — classifies every patch into one of 10 intensity bins using the 99th percentile pixel value. Runs in parallel with Dask threads.
2. `indices_with_complete_sample()` — uses a Numba parallel loop to mark which `(t, i, j)` positions have no missing patches in their full 28-frame window.
3. `starting_indices_for_centers()` — for each bin's patches, finds all `(t, i, j)` positions from which that patch would appear somewhere in the 28-frame window.

At call time `sampler(N)` is O(N) — just random index lookups into pre-built per-bin lists.

---

### `batch.BatchGenerator.batch()` — `batch.py`

Core batch assembly function. For each sample in the batch:

1. Calls `sampler(batch_size)` to get `(t0, i0, j0)` triplets.
2. Computes absolute timestamps for all 28 timesteps.
3. Calls `PatchIndex(t, i0, j0)` which invokes the `build_batch` Numba kernel.
4. Applies the transform closure to get normalized float32 arrays.
5. Optionally applies augmentation (training only).

---

### `batch.PatchIndex.__call__(t, i0, j0)` — `batch.py`

Fast lookup for a batch of `(time, spatial)` positions. Delegates to `build_batch`, a `@njit(parallel=True)` kernel that runs one thread per sample in the batch. Each thread iterates over its 28 timesteps and 4 spatial patches, filling a pre-allocated output buffer.

---

### `batch.StreamBatchDataset` vs `batch.DeterministicBatchDataset` — `batch.py`

| Class | Parent | Behavior |
|-------|--------|----------|
| `StreamBatchDataset` | `IterableDataset` | Calls `batch_gen.batch()` lazily on each `__next__`; infinite stream; different each epoch |
| `DeterministicBatchDataset` | `Dataset` | Pre-samples all `(t, i, j)` positions at construction time with a fixed RNG seed; `__getitem__(i)` always returns the same batch |

---

## Full Function Call Chain

```
setup_data(config)
├── patches.load_all_patches(patch_dir, var)
│   └── [Dask processes] patches.load_patches(fn)  ×N_years
│
├── split.train_valid_test_split(raw, var, ...)
│   └── split.get_chunks(primary_raw, valid_frac, test_frac, chunk_seconds, seed)
│
├── transform.normalize_threshold(log, threshold, fill_value, mean, std)
│   └── returns closure → transform.normalize_threshold_array()  [Numba @njit parallel]
│
└── split.DataModule(variables, raw, ...)
    ├── [×3 splits] batch.BatchGenerator(...)
    │   ├── batch.PatchIndex(patch_data, ...)
    │   │   ├── init_patch_index()        [Numba @njit]
    │   │   └── init_patch_index_zero()   [Numba @njit]
    │   └── sampling.EqualFrequencySampler(bins, patch_data, patch_index, ...)
    │       ├── sampling.bin_classify_patches_parallel()   [Dask threads]
    │       │   └── sampling.bin_classify_patches()  ×N_cores
    │       ├── sampling.indices_with_complete_sample()    [Numba @njit parallel]
    │       └── sampling.starting_indices_for_centers()   [Dask threads + Numba @njit]
    │
    ├── batch.StreamBatchDataset(batch_gen["train"], epoch_size)
    └── batch.DeterministicBatchDataset(batch_gen["valid"/"test"], epoch_size, seed)

─── At training time ───
DataLoader.__iter__()
└── batch.BatchGenerator.batch()
    ├── EqualFrequencySampler.__call__(batch_size)  → (t0, i0, j0) per sample
    ├── PatchIndex.__call__(t, i0, j0)              → raw float32 pixel arrays
    │   └── build_batch()  [Numba @njit parallel — one thread per sample]
    └── transform_fn(raw)                           → normalized float32
```

---

## Dependencies

| Package             | Purpose                                      |
|---------------------|----------------------------------------------|
| `omegaconf`         | YAML configuration loading                   |
| `numpy`             | Array operations                             |
| `netCDF4`           | Reading/writing patch files                  |
| `h5py`              | Reading raw IMERG HDF5 data                  |
| `dask`              | Parallel loading and processing              |
| `numba`             | JIT-compiled batch assembly and sampling     |
| `scipy`             | Antialiasing convolution (transform.py)      |
| `pytorch_lightning` | DataModule and Trainer integration           |
| `torch`             | DataLoader, Dataset, IterableDataset         |
