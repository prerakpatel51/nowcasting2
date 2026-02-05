# Evaluation Metrics

Verification metrics for evaluating precipitation nowcasting forecasts. Both metrics support reading results directly from HDF5 files produced by `run_nowcast.py` and can be used as Python modules or command-line tools.

## Metrics

### CRPS (Continuous Ranked Probability Score)

**File:** `crps.py`

CRPS is a strictly proper scoring rule for evaluating probabilistic (ensemble) forecasts. It measures how well the predicted distribution matches the observed values. For deterministic forecasts, CRPS reduces to MAE.

**Formula (energy form):**

```
CRPS(F, y) = E[|X - y|] - 0.5 * E[|X - X'|]
```

- `X, X'` — independent ensemble members from the forecast distribution
- `y` — observation
- Lower CRPS = better forecast skill

#### Python Usage

```python
from utils.evaluation_metrics.crps import crps_ensemble, evaluate_crps_from_h5

# Direct computation on arrays
# ensemble shape: (n_ensemble, T, H, W), obs shape: (T, H, W)
ensemble = np.random.rand(10, 12, 64, 64)
obs = np.random.rand(12, 64, 64)
crps_values = crps_ensemble(ensemble, obs, axis=0)  # shape: (12, 64, 64)

# Evaluate from an HDF5 file (ensemble mode only)
# Expects 'predictions' (N, E, T, H, W) and 'targets' (N, T, H, W)
crps_values, summary = evaluate_crps_from_h5(
    'path/to/results.h5',
    output_dir='path/to/output',  # optional, defaults to input file directory
    batch_size=100                 # optional, controls memory usage
)

# summary keys: mean_crps, std_crps, mean_crps_per_timestep, mean_crps_spatial,
#               n_samples, n_ensemble, n_timesteps, crps_output_file, summary_output_file
```

#### Command Line

```bash
python -m utils.evaluation_metrics.crps --input results.h5 [--output_dir DIR] [--batch_size 100]
```

#### Output Files

- `crps_values.h5` — Full CRPS array (N, T, H, W) with compression
- `crps_summary.npz` — Summary statistics (mean, std, per-timestep means)

---

### CSI (Critical Success Index)

**File:** `csi.py`

CSI (also called Threat Score) evaluates precipitation forecasts at fixed intensity thresholds by measuring hits vs. false alarms and misses.

**Formula:**

```
CSI = TP / (TP + FP + FN)
```

- TP — both forecast and observation exceed the threshold
- FP — forecast exceeds, observation does not
- FN — observation exceeds, forecast does not
- CSI ranges from 0 (no skill) to 1 (perfect)

For ensemble forecasts, the ensemble mean is computed first, then thresholded.

Default thresholds: `[2, 4, 6, 8, 10]` mm/hr.

#### Python Usage

```python
from utils.evaluation_metrics.csi import csi, evaluate_csi_from_h5

# Direct computation on arrays
# Deterministic forecast
results = csi(predictions, targets, thresholds=[2, 4, 6, 8, 10])
# results = {2: {'csi': 0.85, 'tp': ..., 'fp': ..., 'fn': ...}, ...}

# Ensemble forecast — specify which axis holds ensemble members
results = csi(ensemble_preds, targets, thresholds=[2, 4, 6, 8, 10], ensemble_axis=0)

# Evaluate from an HDF5 file (ensemble or deterministic)
# Ensemble: predictions (N, E, T, H, W), targets (N, T, H, W)
# Deterministic: predictions (N, T, H, W), targets (N, T, H, W)
csi_per_timestep, summary = evaluate_csi_from_h5(
    'path/to/results.h5',
    thresholds=[2, 4, 6, 8, 10],  # optional
    output_dir='path/to/output',   # optional
    batch_size=100                  # optional
)

# csi_per_timestep = {threshold: np.ndarray of shape (T,)}
# summary keys: overall_csi, csi_per_timestep, contingency, thresholds,
#               n_samples, n_timesteps, ensemble_mode, n_ensemble
```

#### Command Line

```bash
# Default thresholds
python -m utils.evaluation_metrics.csi --input results.h5

# Custom thresholds
python -m utils.evaluation_metrics.csi --input results.h5 --thresholds 1 5 10 20 [--output_dir DIR] [--batch_size 100]
```

#### Output Files

- `csi_values.h5` — CSI per timestep and contingency counts per threshold
- `csi_summary.npz` — Summary statistics

---

## Quick Reference

| Metric | Type | Best Value | Input Requirement | Key Property |
|--------|------|------------|-------------------|--------------|
| CRPS | Probabilistic | 0 (lower is better) | Ensemble forecasts only | Generalizes MAE to probabilistic forecasts |
| CSI | Threshold-based | 1 (higher is better) | Ensemble or deterministic | Measures hit rate ignoring correct negatives |

## HDF5 Input Format

Both metrics read HDF5 files with:
- `predictions`: shape `(N, E, T, H, W)` for ensemble or `(N, T, H, W)` for deterministic
- `targets`: shape `(N, T, H, W)`

Where N = samples, E = ensemble members, T = timesteps, H = height, W = width.
