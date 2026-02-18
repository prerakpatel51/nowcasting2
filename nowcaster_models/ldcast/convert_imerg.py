"""Convert IMERG HDF5 data to NetCDF patch format for the ldcast pipeline.

Reads the consolidated IMERG HDF5 file, cuts each 64x64 frame into a
2x2 grid of 32x32 patches, classifies them as data/zero patches, and
saves yearly NetCDF files using the same format as ldcast's patch storage.

Usage:
    python convert_imerg.py
    python convert_imerg.py --config path/to/config.yaml
"""

import argparse
from datetime import datetime
import os

import h5py
import numpy as np
from omegaconf import OmegaConf

from patches import save_patches


def convert_imerg(config):
    h5_path = config.data.h5_path
    patch_size = config.patches.patch_size
    threshold = config.patches.nonzero_threshold
    min_nonzeros = config.patches.min_nonzeros
    image_size = config.patches.image_size
    out_dir = config.data.patch_dir
    var_name = config.data.var_name

    os.makedirs(out_dir, exist_ok=True)

    grid_size = image_size // patch_size  # 64 // 32 = 2

    with h5py.File(h5_path, 'r') as f:
        timestamps = f['timestamps'][:]
        precip = f['precipitation']  # keep as h5 dataset, read per-frame

        # Convert timestamps to datetime years for grouping
        # timestamps are Unix epoch seconds
        dt_years = np.array([
            datetime.utcfromtimestamp(float(t)).year for t in timestamps
        ])

        years = sorted(set(dt_years))
        print(f"Found {len(timestamps)} frames across years {years[0]}-{years[-1]}")

        for year in years:
            year_mask = dt_years == year
            year_indices = np.where(year_mask)[0]
            print(f"\nProcessing {year}: {len(year_indices)} frames")

            patch_data = []
            patch_coords = []
            patch_times = []
            zero_patch_coords = []
            zero_patch_times = []

            for count, idx in enumerate(year_indices):
                if count % 10000 == 0:
                    print(f"  Frame {count}/{len(year_indices)}")

                frame = precip[idx]  # (64, 64)
                t_sec = np.int64(timestamps[idx])

                for pi in range(grid_size):
                    for pj in range(grid_size):
                        i0 = pi * patch_size
                        i1 = i0 + patch_size
                        j0 = pj * patch_size
                        j1 = j0 + patch_size
                        patch = frame[i0:i1, j0:j1].copy()

                        n_nonzero = np.count_nonzero(patch > threshold)

                        if n_nonzero >= min_nonzeros:
                            patch_data.append(patch)
                            patch_coords.append((pi, pj))
                            patch_times.append(t_sec)
                        else:
                            zero_patch_coords.append((pi, pj))
                            zero_patch_times.append(t_sec)

            # Stack into arrays
            if patch_data:
                patch_data_arr = np.stack(patch_data, axis=0)
            else:
                patch_data_arr = np.zeros((0, patch_size, patch_size), dtype=np.float32)
            patch_coords_arr = np.array(patch_coords, dtype=np.uint16).reshape(-1, 2)
            patch_times_arr = np.array(patch_times, dtype=np.int64)

            if zero_patch_coords:
                zero_coords_arr = np.array(zero_patch_coords, dtype=np.uint16).reshape(-1, 2)
                zero_times_arr = np.array(zero_patch_times, dtype=np.int64)
            else:
                zero_coords_arr = np.zeros((0, 2), dtype=np.uint16)
                zero_times_arr = np.zeros((0,), dtype=np.int64)

            out_fn = os.path.join(out_dir, f"patches_{var_name}_{year}.nc")
            save_patches(
                patch_data_arr, patch_coords_arr, patch_times_arr,
                zero_coords_arr, zero_times_arr, out_fn,
                zero_value=0
            )
            print(f"  Saved {out_fn}: {len(patch_data)} data patches, "
                  f"{len(zero_patch_coords)} zero patches")

    print("\nConversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IMERG H5 to NetCDF patches")
    parser.add_argument("--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to config.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    convert_imerg(config)
