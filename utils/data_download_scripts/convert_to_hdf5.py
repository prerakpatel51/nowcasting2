"""
IMERG GeoTIFF to HDF5 Converter

Converts NASA GPM IMERG precipitation GeoTIFF files to HDF5 format optimized
for deep learning dataloaders. Processes files in batches to avoid loading
the entire dataset into memory.

HDF5 Structure:
    /precipitation     - (N, 64, 64) float32 dataset, chunked for efficient access
    /timestamps        - (N,) int64 dataset, Unix timestamps (seconds since epoch)
    /datetime_strings  - (N,) variable-length string dataset, ISO format dates
    /stats/
        /mean          - (N,) float32, mean precipitation per image
        /min           - (N,) float32, min precipitation per image
        /max           - (N,) float32, max precipitation per image
    /metadata          - Group containing dataset attributes:
        - xmin, xmax, ymin, ymax: Geographic bounds
        - height, width: Grid dimensions
        - crs: Coordinate reference system
        - nodata_value: Missing data indicator
        - units: Precipitation units (mm/hr)
        - source: Data source description
        - created: File creation timestamp

Features:
    - Memory-efficient batch processing
    - Chronologically sorted output
    - HDF5 chunking optimized for sequential and random access
    - LZF compression for fast read/write with good compression
    - Per-image statistics (mean, min, max) for quick filtering
    - Gap detection via timestamps

Configuration:
    All paths are read from config.py. Output location and batch size
    can be overridden via environment variables.

Environment Variables (optional overrides):
    HDF5_OUTPUT_DIR: Output directory for HDF5 file
    HDF5_BATCH_SIZE: Number of files to process per batch (default: 1000)

Usage:
    python convert_to_hdf5.py

Output:
    Creates imerg_data.h5 in the configured output directory.
"""

import os
import sys
import glob
import datetime
from typing import List, Tuple, Optional
import numpy as np
import h5py
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly

from config import (
    IMERG_DATA_DIR,
    XMIN, XMAX, YMIN, YMAX,
    OUTPUT_HEIGHT, OUTPUT_WIDTH,
    HDF5_OUTPUT_DIR as CONFIG_HDF5_OUTPUT_DIR,
    HDF5_FILENAME as CONFIG_HDF5_FILENAME,
    HDF5_BATCH_SIZE as CONFIG_HDF5_BATCH_SIZE,
    HDF5_TEMPORAL_CHUNK_SIZE,
    HDF5_COMPRESSION,
    HDF5_NODATA_VALUE,
)

# =============================================================================
# CONFIGURATION - Environment variables override config.py defaults
# =============================================================================

HDF5_OUTPUT_DIR = os.environ.get('HDF5_OUTPUT_DIR', CONFIG_HDF5_OUTPUT_DIR)
HDF5_FILENAME = os.environ.get('HDF5_FILENAME', CONFIG_HDF5_FILENAME)
BATCH_SIZE = int(os.environ.get('HDF5_BATCH_SIZE', CONFIG_HDF5_BATCH_SIZE))
TEMPORAL_CHUNK_SIZE = HDF5_TEMPORAL_CHUNK_SIZE
COMPRESSION = HDF5_COMPRESSION
NODATA_VALUE = HDF5_NODATA_VALUE


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_filename_timestamp(filename: str) -> Optional[datetime.datetime]:
    """
    Extract datetime from IMERG filename.

    IMERG filenames use UTC time. The returned datetime is timezone-aware (UTC).

    Args:
        filename: Filename in format 'imerg.YYYYMMDDHHMM.tif'

    Returns:
        Parsed datetime object (UTC timezone-aware), or None if parsing fails.
    """
    basename = os.path.basename(filename)
    try:
        parts = basename.split('.')
        if len(parts) >= 2 and parts[0] == 'imerg':
            timestamp_str = parts[1]
            # Parse as naive datetime then make it UTC-aware
            naive_dt = datetime.datetime.strptime(timestamp_str, '%Y%m%d%H%M')
            return naive_dt.replace(tzinfo=datetime.timezone.utc)
    except (ValueError, IndexError):
        pass
    return None


def read_geotiff(filepath: str) -> Tuple[Optional[np.ndarray], dict]:
    """
    Read precipitation data from a GeoTIFF file.

    Args:
        filepath: Path to the GeoTIFF file.

    Returns:
        Tuple of (data_array, metadata_dict).
        data_array is None if file cannot be read.
    """
    metadata = {}

    try:
        ds = gdal.Open(filepath, GA_ReadOnly)
        if ds is None:
            return None, metadata

        band = ds.GetRasterBand(1)
        data = band.ReadAsArray().astype(np.float32)

        metadata = {
            'geotransform': ds.GetGeoTransform(),
            'projection': ds.GetProjection(),
            'nodata': band.GetNoDataValue(),
            'width': ds.RasterXSize,
            'height': ds.RasterYSize
        }

        ds = None
        return data, metadata

    except Exception as e:
        print(f"  Warning: Error reading {filepath}: {e}")
        return None, metadata


def get_sorted_file_list(data_dir: str) -> List[Tuple[str, datetime.datetime]]:
    """
    Get list of IMERG files sorted by timestamp.

    Args:
        data_dir: Directory containing IMERG GeoTIFF files.

    Returns:
        List of (filepath, datetime) tuples, sorted chronologically.
    """
    pattern = os.path.join(data_dir, 'imerg.*.tif')
    files = glob.glob(pattern)

    file_list = []
    for filepath in files:
        timestamp = parse_filename_timestamp(filepath)
        if timestamp is not None:
            file_list.append((filepath, timestamp))

    file_list.sort(key=lambda x: x[1])
    return file_list


def create_hdf5_file(output_path: str, num_samples: int, height: int, width: int) -> h5py.File:
    """
    Create HDF5 file with pre-allocated datasets.

    Args:
        output_path: Path for the output HDF5 file.
        num_samples: Total number of time samples.
        height, width: Spatial dimensions.

    Returns:
        Open HDF5 file handle (must be closed by caller).
    """
    f = h5py.File(output_path, 'w')

    # Precipitation data: (N, H, W)
    chunk_shape = (min(TEMPORAL_CHUNK_SIZE, num_samples), height, width)
    f.create_dataset(
        'precipitation',
        shape=(num_samples, height, width),
        dtype=np.float32,
        chunks=chunk_shape,
        compression=COMPRESSION,
        fillvalue=NODATA_VALUE
    )

    # Timestamps as Unix epoch (int64)
    time_chunk = min(TEMPORAL_CHUNK_SIZE * 10, num_samples)
    f.create_dataset(
        'timestamps',
        shape=(num_samples,),
        dtype=np.int64,
        chunks=(time_chunk,),
        compression=COMPRESSION
    )

    # Datetime strings for human readability
    dt_string = h5py.special_dtype(vlen=str)
    f.create_dataset(
        'datetime_strings',
        shape=(num_samples,),
        dtype=dt_string,
        chunks=(time_chunk,)
    )

    # Statistics group
    stats = f.create_group('stats')
    for stat_name in ['mean', 'min', 'max']:
        stats.create_dataset(
            stat_name,
            shape=(num_samples,),
            dtype=np.float32,
            chunks=(time_chunk,),
            compression=COMPRESSION,
            fillvalue=np.nan
        )

    # Metadata group
    meta = f.create_group('metadata')
    meta.attrs['xmin'] = XMIN
    meta.attrs['xmax'] = XMAX
    meta.attrs['ymin'] = YMIN
    meta.attrs['ymax'] = YMAX
    meta.attrs['height'] = height
    meta.attrs['width'] = width
    meta.attrs['crs'] = 'EPSG:4326'
    meta.attrs['nodata_value'] = NODATA_VALUE
    meta.attrs['units'] = 'mm/hr'
    meta.attrs['source'] = 'NASA GPM IMERG Early Run (3B-HHR-E)'
    meta.attrs['timezone'] = 'UTC'
    meta.attrs['temporal_resolution_minutes'] = 30
    meta.attrs['spatial_resolution'] = f'{(XMAX-XMIN)/width:.4f} x {(YMAX-YMIN)/height:.4f} degrees'
    meta.attrs['created'] = datetime.datetime.now().isoformat()
    meta.attrs['num_samples'] = num_samples

    return f


def compute_stats(data: np.ndarray, nodata: float) -> Tuple[float, float, float]:
    """
    Compute mean, min, max of an image, ignoring nodata values.

    Args:
        data: 2D precipitation array.
        nodata: NoData value to ignore.

    Returns:
        Tuple of (mean, min, max). Returns (nan, nan, nan) if all nodata.
    """
    valid_mask = data != nodata
    if not np.any(valid_mask):
        return np.nan, np.nan, np.nan

    valid_data = data[valid_mask]
    return float(np.mean(valid_data)), float(np.min(valid_data)), float(np.max(valid_data))


def process_batch(
    hdf5_file: h5py.File,
    file_batch: List[Tuple[str, datetime.datetime]],
    start_idx: int,
    height: int,
    width: int
) -> Tuple[int, int]:
    """
    Process a batch of GeoTIFF files and write to HDF5.

    Args:
        hdf5_file: Open HDF5 file handle.
        file_batch: List of (filepath, datetime) tuples to process.
        start_idx: Starting index in the HDF5 datasets.
        height, width: Expected spatial dimensions.

    Returns:
        Tuple of (successful_count, failed_count).
    """
    batch_size = len(file_batch)

    # Pre-allocate batch arrays
    precip_batch = np.full((batch_size, height, width), NODATA_VALUE, dtype=np.float32)
    timestamps_batch = np.zeros(batch_size, dtype=np.int64)
    datetime_strings_batch = []
    mean_batch = np.full(batch_size, np.nan, dtype=np.float32)
    min_batch = np.full(batch_size, np.nan, dtype=np.float32)
    max_batch = np.full(batch_size, np.nan, dtype=np.float32)

    successful = 0
    failed = 0

    for i, (filepath, timestamp) in enumerate(file_batch):
        data, _ = read_geotiff(filepath)

        timestamps_batch[i] = int(timestamp.timestamp())
        datetime_strings_batch.append(timestamp.isoformat())

        if data is not None and data.shape == (height, width):
            precip_batch[i] = data
            mean_batch[i], min_batch[i], max_batch[i] = compute_stats(data, NODATA_VALUE)
            successful += 1
        else:
            failed += 1

    # Write batch to HDF5
    end_idx = start_idx + batch_size
    hdf5_file['precipitation'][start_idx:end_idx] = precip_batch
    hdf5_file['timestamps'][start_idx:end_idx] = timestamps_batch
    hdf5_file['datetime_strings'][start_idx:end_idx] = datetime_strings_batch
    hdf5_file['stats/mean'][start_idx:end_idx] = mean_batch
    hdf5_file['stats/min'][start_idx:end_idx] = min_batch
    hdf5_file['stats/max'][start_idx:end_idx] = max_batch

    hdf5_file.flush()

    return successful, failed


def convert_to_hdf5():
    """
    Main conversion function.

    Scans the IMERG data directory, processes files in batches,
    and writes to HDF5 format.
    """
    print("=" * 70)
    print("IMERG GeoTIFF to HDF5 Converter")
    print("=" * 70)

    os.makedirs(HDF5_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(HDF5_OUTPUT_DIR, HDF5_FILENAME)

    print(f"Input directory:  {IMERG_DATA_DIR}")
    print(f"Output file:      {output_path}")
    print(f"Batch size:       {BATCH_SIZE}")
    print("-" * 70)

    # Get sorted file list
    print("Scanning for IMERG files...")
    file_list = get_sorted_file_list(IMERG_DATA_DIR)
    num_files = len(file_list)

    if num_files == 0:
        print("Error: No IMERG files found!")
        sys.exit(1)

    print(f"Found {num_files:,} files")
    print(f"Date range: {file_list[0][1]} to {file_list[-1][1]}")
    print("-" * 70)

    # Create HDF5 file
    print("Creating HDF5 file...")
    hdf5_file = create_hdf5_file(output_path, num_files, OUTPUT_HEIGHT, OUTPUT_WIDTH)

    # Process in batches
    total_successful = 0
    total_failed = 0
    num_batches = (num_files + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Processing {num_batches} batches...")
    print("-" * 70)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, num_files)
        file_batch = file_list[start_idx:end_idx]

        progress = (batch_idx + 1) / num_batches * 100
        print(f"Batch {batch_idx + 1}/{num_batches} "
              f"({start_idx:,}-{end_idx-1:,}) ... ", end='', flush=True)

        successful, failed = process_batch(
            hdf5_file, file_batch, start_idx, OUTPUT_HEIGHT, OUTPUT_WIDTH
        )

        total_successful += successful
        total_failed += failed
        print(f"OK ({successful} success, {failed} failed) [{progress:.1f}%]")

    # Update metadata with final counts
    hdf5_file['metadata'].attrs['successful_reads'] = total_successful
    hdf5_file['metadata'].attrs['failed_reads'] = total_failed

    # Detect and store temporal gaps
    print("-" * 70)
    print("Detecting temporal gaps...")
    timestamps = hdf5_file['timestamps'][:]
    expected_gap = 30 * 60  # 30 minutes in seconds
    gaps = []
    for i in range(len(timestamps) - 1):
        diff = timestamps[i + 1] - timestamps[i]
        if diff != expected_gap:
            gaps.append((i, int(diff)))

    if gaps:
        print(f"Found {len(gaps)} gaps in the data")
        # Store gap info as dataset
        gap_indices = np.array([g[0] for g in gaps], dtype=np.int32)
        gap_durations = np.array([g[1] for g in gaps], dtype=np.int32)
        hdf5_file.create_dataset('gaps/indices', data=gap_indices)
        hdf5_file.create_dataset('gaps/durations_seconds', data=gap_durations)
    else:
        print("No temporal gaps detected - data is continuous")

    hdf5_file.close()

    # Print summary
    print("=" * 70)
    print("Conversion Complete!")
    print("=" * 70)
    print(f"Output file:       {output_path}")
    print(f"Total samples:     {num_files:,}")
    print(f"Successful reads:  {total_successful:,}")
    print(f"Failed reads:      {total_failed:,}")
    print(f"Temporal gaps:     {len(gaps)}")

    file_size = os.path.getsize(output_path)
    if file_size > 1e9:
        print(f"File size:         {file_size / 1e9:.2f} GB")
    else:
        print(f"File size:         {file_size / 1e6:.2f} MB")

    print("=" * 70)


if __name__ == "__main__":
    convert_to_hdf5()
