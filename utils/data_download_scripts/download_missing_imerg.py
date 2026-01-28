"""
Download Missing IMERG Files

Identifies and downloads missing NASA GPM IMERG precipitation data files
for a specified date range. Scans an existing data directory to find gaps
in the time series and optionally downloads the missing files from NASA PPS.

This is a companion script to download_imerg_data.py, designed for data
maintenance and gap-filling after the initial bulk download.

Features:
    - Scans existing data directory for missing 30-minute time steps
    - Dry-run mode to preview missing files without downloading
    - Progress tracking with success/failure counts
    - Same processing pipeline as download_imerg_data.py

Configuration:
    All configuration is centralized in config.py. Modify config.py to
    change date ranges, paths, and credentials.

Usage:
    python download_missing_imerg.py

Data Source: NASA Precipitation Processing System (PPS)
"""

import os
import subprocess
import datetime
from datetime import timedelta

import osgeo.gdal as gdal
from osgeo.gdalconst import GA_ReadOnly

# Import configuration from centralized config file
from config import (
    XMIN, XMAX, YMIN, YMAX,
    OUTPUT_HEIGHT, OUTPUT_WIDTH,
    MISSING_CHECK_START_DATE,
    MISSING_CHECK_END_DATE,
    EMAIL, SERVER_PATH,
    IMERG_DATA_DIR,
    DRY_RUN,
    SCRIPT_DIR,
    ensure_directories
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Date range and folders from config
START_DATE = MISSING_CHECK_START_DATE
END_DATE = MISSING_CHECK_END_DATE
IMERG_DATA_FOLDER = IMERG_DATA_DIR
OUTPUT_FOLDER = IMERG_DATA_DIR

# =============================================================================
# FUNCTIONS
# =============================================================================


def find_missing_files(start_date, end_date, data_folder):
    """
    Scan a data directory and identify missing IMERG files within a date range.

    Args:
        start_date: Start of the date range to check (inclusive).
        end_date: End of the date range to check (inclusive).
        data_folder: Path to the directory containing existing IMERG files.

    Returns:
        List of datetime objects for timestamps where files are missing.
    """
    missing = []
    current = start_date
    delta = timedelta(minutes=30)

    while current <= end_date:
        filename = f"imerg.{current.strftime('%Y%m%d%H%M')}.tif"
        filepath = os.path.join(data_folder, filename)

        if not os.path.isfile(filepath):
            missing.append(current)
        current += delta

    return missing


def get_file(filename, server, email):
    """
    Download a file from the NASA PPS server using curl.

    Args:
        filename: Relative path to the file on the server.
        server: Base URL of the NASA PPS server.
        email: NASA Earthdata registered email for authentication.

    Returns:
        True if file downloaded successfully, False otherwise.
    """
    url = server + '/' + filename
    local_name = os.path.basename(filename)

    cmd = f'curl -sO -u {email}:{email} {url}'
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()

    if os.path.isfile(local_name):
        with open(local_name, 'rb') as f:
            header = f.read(100)
            if b'<!DOCTYPE' in header or b'<html' in header.lower():
                os.remove(local_name)
                return False
    return True


def read_and_warp(grid_file, xmin, ymin, xmax, ymax, req_height, req_width):
    """
    Read a GeoTIFF and warp it to a target geographic extent and resolution.

    Args:
        grid_file: Path to the input GeoTIFF file.
        xmin, ymin, xmax, ymax: Geographic bounds (decimal degrees).
        req_height, req_width: Output grid dimensions in pixels.

    Returns:
        Tuple of (data_array, width, height, geotransform, projection).

    Raises:
        RuntimeError: If the input file cannot be opened.
    """
    raw_grid = gdal.Open(grid_file, GA_ReadOnly)
    if raw_grid is None:
        raise RuntimeError(f"Failed to open {grid_file}")

    pre_ds = gdal.Translate('temp.tif', raw_grid,
                            options="-co COMPRESS=Deflate -a_nodata 29999 -a_ullr -180.0 90.0 180.0 -90.0")

    pixel_size_x = (xmax - xmin) / req_width
    pixel_size_y = (ymax - ymin) / req_height

    ds = gdal.Warp('', pre_ds, srcNodata=29999, srcSRS='EPSG:4326', dstSRS='EPSG:4326',
                   dstNodata='-9999', format='VRT', xRes=pixel_size_x, yRes=-pixel_size_y,
                   outputBounds=(xmin, ymin, xmax, ymax))

    return ds.ReadAsArray(), ds.GetRasterBand(1).XSize, ds.GetRasterBand(1).YSize, \
           ds.GetGeoTransform(), ds.GetProjection()


def write_grid(grid_out_name, data_out, nx, ny, gt, proj):
    """
    Write precipitation data to a compressed GeoTIFF file.

    Args:
        grid_out_name: Output file path.
        data_out: 2D array of precipitation values (mm/hr).
        nx, ny: Output dimensions in pixels.
        gt: GDAL GeoTransform tuple.
        proj: WKT projection string.
    """
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(grid_out_name, nx, ny, 1, gdal.GDT_Float32, ['COMPRESS=DEFLATE'])

    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)

    data_out.shape = (-1, nx)
    dst_ds.GetRasterBand(1).WriteArray(data_out, 0, 0)
    dst_ds.GetRasterBand(1).SetNoDataValue(-9999.0)

    dst_ds = None


def download_single(date, output_folder):
    """
    Download and process a single missing IMERG file.

    Args:
        date: Target timestamp for the output file (end of 30-min period).
        output_folder: Directory to save the processed GeoTIFF.

    Returns:
        True if download and processing succeeded, False otherwise.
    """
    file_prefix = '3B-HHR-E.MS.MRG.3IMERG.'
    file_suffix = '.V07B.30min.tif'

    download_date = date - timedelta(minutes=30)

    initial_time_stmp = download_date.strftime('%Y%m%d-S%H%M%S')
    final_time_stmp = (download_date + timedelta(minutes=29)).strftime('E%H%M59')
    folder = download_date.strftime('%Y/%m/')
    total_minutes = download_date.hour * 60 + download_date.minute
    date_stamp = f"{initial_time_stmp}-{final_time_stmp}.{total_minutes:04}"

    remote_filename = folder + file_prefix + date_stamp + file_suffix
    local_filename = file_prefix + date_stamp + file_suffix
    grid_out_name = os.path.join(output_folder, f"imerg.{date.strftime('%Y%m%d%H%M')}.tif")

    get_file(remote_filename, SERVER_PATH, EMAIL)

    if os.path.isfile(local_filename):
        try:
            new_grid, nx, ny, gt, proj = read_and_warp(
                local_filename, XMIN, YMIN, XMAX, YMAX, OUTPUT_HEIGHT, OUTPUT_WIDTH
            )
            new_grid = new_grid * 0.1
            write_grid(grid_out_name, new_grid, nx, ny, gt, proj)
            return True
        except Exception as e:
            print(f"  Error: {e}")
            return False
        finally:
            if os.path.isfile(local_filename):
                os.remove(local_filename)
            if os.path.isfile('temp.tif'):
                os.remove('temp.tif')
    return False


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Change to script directory for temp file handling
    original_dir = os.getcwd()
    os.chdir(SCRIPT_DIR)

    try:
        print("=" * 60)
        print("IMERG Missing File Checker & Downloader")
        print("=" * 60)

        print(f"\nChecking for missing files from {START_DATE} to {END_DATE}...")
        missing = find_missing_files(START_DATE, END_DATE, IMERG_DATA_FOLDER)

        print(f"\nFound {len(missing)} missing files")

        if not missing:
            print("All files present!")
            exit(0)

        print("\nMissing files:")
        for i, d in enumerate(missing):
            print(f"  {i+1}. {d.strftime('%Y-%m-%d %H:%M')} -> imerg.{d.strftime('%Y%m%d%H%M')}.tif")

        print(f"\nTotal: {len(missing)} missing files")

        if DRY_RUN:
            print("\n" + "=" * 60)
            print("DRY RUN MODE - No files downloaded")
            print("To download, set DRY_RUN = False in config.py")
            print("=" * 60)
            exit(0)

        print(f"\nDownloading {len(missing)} missing files...")
        ensure_directories()

        success = 0
        failed = 0

        for i, date in enumerate(missing):
            print(f"[{i+1}/{len(missing)}] Downloading {date.strftime('%Y-%m-%d %H:%M')}", end=" ")
            if download_single(date, OUTPUT_FOLDER):
                print("OK")
                success += 1
            else:
                print("FAILED")
                failed += 1

        print(f"\nDone! Success: {success}, Failed: {failed}")

    finally:
        os.chdir(original_dir)
