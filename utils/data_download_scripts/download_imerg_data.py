"""
IMERG Data Downloader

Downloads NASA GPM IMERG (Integrated Multi-satellitE Retrievals for GPM)
precipitation data for a specified geographic region and time period.
Downloaded data is processed, warped to a target grid, and saved as GeoTIFF files.

IMERG provides global precipitation estimates at 0.1 x 0.1 degree resolution with
30-minute temporal resolution. This script downloads the Early Run product
(3B-HHR-E), available with approximately 4 hour latency.

Features:
    - Downloads IMERG Early Run half-hourly precipitation data
    - Automatically clips to user-defined geographic bounds
    - Resamples to user-specified output grid dimensions
    - Handles authentication with NASA PPS (Precipitation Processing System)
    - Skips already-downloaded files to support resumable downloads
    - Cleans up temporary files automatically

Configuration:
    All configuration is centralized in config.py. Parameters can be overridden
    via environment variables for batch processing with SLURM.

Environment Variables (optional overrides):
    XMIN, XMAX, YMIN, YMAX: Geographic bounds (longitude/latitude)
    OUTPUT_HEIGHT, OUTPUT_WIDTH: Output grid dimensions in pixels
    START_DATE, END_DATE: Time period in format 'YYYY-MM-DD-HH-MM'
    OUTPUT_FOLDER: Directory to save processed GeoTIFF files
    EMAIL: NASA Earthdata registered email for authentication
    SERVER_PATH: NASA PPS server URL

Usage:
    python download_imerg_data.py

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
    XMIN as CONFIG_XMIN,
    XMAX as CONFIG_XMAX,
    YMIN as CONFIG_YMIN,
    YMAX as CONFIG_YMAX,
    OUTPUT_HEIGHT as CONFIG_OUTPUT_HEIGHT,
    OUTPUT_WIDTH as CONFIG_OUTPUT_WIDTH,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    EMAIL as CONFIG_EMAIL,
    SERVER_PATH as CONFIG_SERVER_PATH,
    get_output_folder,
    SCRIPT_DIR
)

# =============================================================================
# CONFIGURATION - Environment variables override config.py defaults
# =============================================================================

# Geographic bounding box (decimal degrees, WGS84)
XMIN = float(os.environ.get('XMIN', CONFIG_XMIN))
XMAX = float(os.environ.get('XMAX', CONFIG_XMAX))
YMIN = float(os.environ.get('YMIN', CONFIG_YMIN))
YMAX = float(os.environ.get('YMAX', CONFIG_YMAX))

# Output grid dimensions in pixels
OUTPUT_HEIGHT = int(os.environ.get('OUTPUT_HEIGHT', CONFIG_OUTPUT_HEIGHT))
OUTPUT_WIDTH = int(os.environ.get('OUTPUT_WIDTH', CONFIG_OUTPUT_WIDTH))


def parse_date(date_str, default):
    """
    Parse a date string into a datetime object.

    Args:
        date_str: Date string in format 'YYYY-MM-DD-HH-MM' or None.
        default: Default datetime to return if date_str is None/empty.

    Returns:
        Parsed datetime object or default value.
    """
    if date_str:
        parts = date_str.split('-')
        return datetime.datetime(int(parts[0]), int(parts[1]), int(parts[2]),
                                 int(parts[3]), int(parts[4]), 0)
    return default


# Time period for download
START_DATE = parse_date(os.environ.get('START_DATE'), DEFAULT_START_DATE)
END_DATE = parse_date(os.environ.get('END_DATE'), DEFAULT_END_DATE)

# Output directory
OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', get_output_folder())

# NASA PPS credentials and server
EMAIL = os.environ.get('EMAIL', CONFIG_EMAIL)
SERVER_PATH = os.environ.get('SERVER_PATH', CONFIG_SERVER_PATH)

# =============================================================================
# FUNCTIONS
# =============================================================================


def get_file(filename, server, email):
    """
    Download a file from the NASA PPS server.

    Uses curl with HTTP Basic Authentication. NASA PPS uses the same email
    address for both username and password. Validates that the downloaded
    file is not an HTML error page.

    Args:
        filename: Relative path to the file on the server.
        server: Base URL of the NASA PPS server.
        email: NASA Earthdata registered email for authentication.

    Returns:
        True if download succeeded and file is valid, False otherwise.
    """
    url = server + '/' + filename
    local_name = os.path.basename(filename)

    cmd = f'curl -sO -u {email}:{email} {url}'
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()

    # Check if downloaded file is an HTML error page
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

    Processes raw IMERG files by:
    1. Setting correct global georeference (-180 to 180, -90 to 90)
    2. Clipping to the region of interest
    3. Resampling to the target grid dimensions

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
        raise RuntimeError(f"Failed to open {grid_file} - file may be corrupted")

    # Set correct global extent and no-data value
    pre_ds = gdal.Translate('temp.tif', raw_grid,
                            options="-co COMPRESS=Deflate -a_nodata 29999 -a_ullr -180.0 90.0 180.0 -90.0")

    # Calculate pixel sizes for target dimensions
    pixel_size_x = (xmax - xmin) / req_width
    pixel_size_y = (ymax - ymin) / req_height

    # Warp to target extent and resolution
    ds = gdal.Warp('', pre_ds, srcNodata=29999, srcSRS='EPSG:4326', dstSRS='EPSG:4326',
                   dstNodata='-9999', format='VRT', xRes=pixel_size_x, yRes=-pixel_size_y,
                   outputBounds=(xmin, ymin, xmax, ymax))

    warped_grid = ds.ReadAsArray()
    new_gt = ds.GetGeoTransform()
    new_proj = ds.GetProjection()
    new_nx = ds.GetRasterBand(1).XSize
    new_ny = ds.GetRasterBand(1).YSize

    return warped_grid, new_nx, new_ny, new_gt, new_proj


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


def download_imerg(output_folder, start_date, end_date, xmin, ymin, xmax, ymax,
                   req_height, req_width, server, email):
    """
    Download and process IMERG data for a specified region and time period.

    Iterates through each 30-minute time step, downloads the corresponding
    IMERG file, processes it to extract the region of interest, and saves
    the result as a GeoTIFF.

    Args:
        output_folder: Directory to save processed GeoTIFF files.
        start_date, end_date: Time period for download.
        xmin, ymin, xmax, ymax: Geographic bounds (decimal degrees).
        req_height, req_width: Output grid dimensions in pixels.
        server: NASA PPS server URL.
        email: NASA Earthdata email for authentication.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Change to script directory for temp file handling
    original_dir = os.getcwd()
    os.chdir(SCRIPT_DIR)

    try:
        file_prefix = '3B-HHR-E.MS.MRG.3IMERG.'
        file_suffix = '.V07B.30min.tif'
        delta_time = timedelta(minutes=30)

        current_date = start_date
        final_date = end_date + timedelta(minutes=30)

        while current_date < final_date:
            initial_time_stmp = current_date.strftime('%Y%m%d-S%H%M%S')
            final_time = current_date + timedelta(minutes=29)
            final_time_stmp = final_time.strftime('E%H%M59')
            final_time_gridout = current_date + timedelta(minutes=30)

            folder = current_date.strftime('%Y/%m/')
            total_minutes = current_date.hour * 60 + current_date.minute
            date_stamp = f"{initial_time_stmp}-{final_time_stmp}.{total_minutes:04}"

            filename = folder + file_prefix + date_stamp + file_suffix
            local_filename = file_prefix + date_stamp + file_suffix
            grid_out_name = os.path.join(output_folder, 'imerg.' + final_time_gridout.strftime('%Y%m%d%H%M') + '.tif')

            print(f'Downloading {final_time_gridout.strftime("%Y-%m-%d %H:%M")}')

            if not os.path.isfile(grid_out_name):
                get_file(filename, server, email)

                if os.path.isfile(local_filename):
                    try:
                        new_grid, nx, ny, gt, proj = read_and_warp(
                            local_filename, xmin, ymin, xmax, ymax, req_height, req_width
                        )
                        new_grid = new_grid * 0.1  # Scale to mm/hr
                        write_grid(grid_out_name, new_grid, nx, ny, gt, proj)
                    except RuntimeError as e:
                        print(f'  Warning: {e}')
                    finally:
                        if os.path.isfile(local_filename):
                            os.remove(local_filename)
                        if os.path.isfile('temp.tif'):
                            os.remove('temp.tif')
                else:
                    print(f'  Warning: Failed to download {local_filename}')
            else:
                print(f'  File already exists, skipping')

            current_date += delta_time

        print(f'\nDownload complete! Files saved to: {output_folder}')

    finally:
        os.chdir(original_dir)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("IMERG Data Downloader")
    print("=" * 60)
    print(f"Region: [{XMIN}, {XMAX}] x [{YMIN}, {YMAX}]")
    print(f"Grid size: {OUTPUT_WIDTH} x {OUTPUT_HEIGHT}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Output: {OUTPUT_FOLDER}")
    print("=" * 60)

    download_imerg(
        output_folder=OUTPUT_FOLDER,
        start_date=START_DATE,
        end_date=END_DATE,
        xmin=XMIN,
        ymin=YMIN,
        xmax=XMAX,
        ymax=YMAX,
        req_height=OUTPUT_HEIGHT,
        req_width=OUTPUT_WIDTH,
        server=SERVER_PATH,
        email=EMAIL
    )
