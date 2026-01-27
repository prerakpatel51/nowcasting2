"""
Download Missing IMERG Files

This script identifies and downloads missing NASA GPM IMERG precipitation data files
for a specified date range. It scans an existing data directory to find gaps in the
time series and optionally downloads the missing files from NASA PPS.

This is a companion script to download_imerg_data.py, designed for data maintenance
and gap-filling after the initial bulk download. It's particularly useful for:
- Recovering from interrupted downloads
- Filling gaps caused by temporary server unavailability
- Validating data completeness before running analysis pipelines

Features:
    - Scans existing data directory for missing 30-minute time steps
    - Dry-run mode (default) to preview missing files without downloading
    - Progress tracking with success/failure counts
    - Same processing pipeline as download_imerg_data.py for consistency

Workflow:
    1. Run with DRY_RUN=True (default) to identify missing files
    2. Review the list of missing files
    3. Set DRY_RUN=False and re-run to download missing data

Requirements:
    - NASA Earthdata account (register at https://urs.earthdata.nasa.gov/)
    - GDAL Python bindings (osgeo)
    - curl command-line tool
    - Existing IMERG data directory (from download_imerg_data.py)

Usage:
    1. Edit configuration section below (date range, folders, credentials)
    2. Run with DRY_RUN=True to see missing files
    3. Set DRY_RUN=False and run again to download

Example:
    $ python download_missing_imerg.py  # Lists missing files (dry run)
    # Edit script: DRY_RUN = False
    $ python download_missing_imerg.py  # Downloads missing files

Data Source: NASA Precipitation Processing System (PPS)
"""

import os
import subprocess
import datetime
from datetime import timedelta

# GDAL is used for reading, warping, and writing geospatial raster data
import osgeo.gdal as gdal
from osgeo.gdalconst import GA_ReadOnly

# =============================================================================
# CONFIGURATION
# =============================================================================
# Modify these settings to match your data requirements and credentials.

# Dry run mode toggle
# When True: Only lists missing files without downloading (safe for testing)
# When False: Actually downloads and processes missing files
DRY_RUN = False  # <-- CHANGE TO False TO ENABLE DOWNLOADING

# Date range to check for missing files
# The script will check every 30-minute interval in this range
# Note: START_DATE should use 0:30 to align with IMERG output file naming
# (IMERG files are named by their END time, so 0:30 is the first file of each day)
START_DATE = datetime.datetime(2011, 4, 1, 0, 30, 0)   # First timestamp to check
END_DATE = datetime.datetime(2022, 12, 31, 23, 30, 0)  # Last timestamp to check

# Data directories
# IMERG_DATA_FOLDER: Existing directory to scan for gaps
# OUTPUT_FOLDER: Where to save downloaded files (usually same as IMERG_DATA_FOLDER)
IMERG_DATA_FOLDER = '/home1/ppatel2025/ldcast/data/imerg_data/'
OUTPUT_FOLDER = IMERG_DATA_FOLDER

# Geographic bounding box for the region of interest
# Coordinates are in decimal degrees (WGS84 / EPSG:4326)
# These should match the settings used in download_imerg_data.py
XMIN, XMAX = -5.5, 2.5    # Western and Eastern longitude bounds (Burkina Faso)
YMIN, YMAX = 9.0, 17.0    # Southern and Northern latitude bounds
OUTPUT_HEIGHT, OUTPUT_WIDTH = 64, 64  # Output grid dimensions in pixels

# NASA PPS authentication credentials
# Register at https://urs.earthdata.nasa.gov/ to get your credentials
EMAIL = 'aaravamudan2014@my.fit.edu'

# NASA PPS IMERG data server URL
# 'early/' = Early Run (~4hr latency), 'late/' = Late Run, 'final/' = Final Run
SERVER_PATH = 'https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/'

# =============================================================================
# FUNCTIONS
# =============================================================================

def find_missing_files(start_date, end_date, data_folder):
    """
    Scan a data directory and identify missing IMERG files within a date range.

    Iterates through every 30-minute time step from start_date to end_date
    and checks if the corresponding IMERG GeoTIFF file exists. Returns a list
    of datetime objects for timestamps where files are missing.

    The expected filename format is: imerg.YYYYMMDDHHMM.tif
    (e.g., imerg.202106011230.tif for June 1, 2021 at 12:30)

    Args:
        start_date (datetime.datetime): Start of the date range to check (inclusive).
            Should typically be at :00 or :30 minutes to align with IMERG timing.
        end_date (datetime.datetime): End of the date range to check (inclusive).
        data_folder (str): Path to the directory containing existing IMERG files.

    Returns:
        list[datetime.datetime]: List of datetime objects representing timestamps
            for which IMERG files are missing. Empty list if all files are present.

    Example:
        >>> missing = find_missing_files(
        ...     datetime.datetime(2021, 6, 1, 0, 30),
        ...     datetime.datetime(2021, 6, 1, 2, 0),
        ...     '/data/imerg/'
        ... )
        >>> len(missing)  # Number of missing 30-minute intervals
        3
    """
    missing = []
    current = start_date
    delta = timedelta(minutes=30)  # IMERG temporal resolution

    # Iterate through each 30-minute time step
    while current <= end_date:
        # Construct expected filename based on timestamp
        filename = f"imerg.{current.strftime('%Y%m%d%H%M')}.tif"
        filepath = os.path.join(data_folder, filename)

        # Check if file exists and add to missing list if not
        if not os.path.isfile(filepath):
            missing.append(current)
        current += delta

    return missing

def get_file(filename, server, email):
    """
    Download a file from the NASA PPS server using curl.

    Uses HTTP Basic Authentication with the NASA PPS server. The server
    uses the same email address for both username and password. Also
    validates that the downloaded file is not an HTML error page.

    Args:
        filename (str): Relative path to the file on the server, including
            subdirectories (e.g., '2021/06/3B-HHR-E.MS.MRG.3IMERG...tif').
        server (str): Base URL of the NASA PPS server.
        email (str): NASA Earthdata registered email for authentication.

    Returns:
        bool: True if file downloaded successfully and is valid (not HTML).
            False if download failed or received an HTML error page.

    Side Effects:
        - Creates a file in the current working directory.
        - Removes the file if it's detected as HTML (error page).
    """
    # Build full URL and extract local filename
    url = server + '/' + filename
    local_name = os.path.basename(filename)

    # Download using curl with silent mode and HTTP Basic Auth
    cmd = f'curl -sO -u {email}:{email} {url}'
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()

    # Validate downloaded file isn't an HTML error page
    if os.path.isfile(local_name):
        with open(local_name, 'rb') as f:
            header = f.read(100)
            if b'<!DOCTYPE' in header or b'<html' in header.lower():
                os.remove(local_name)  # Remove invalid HTML file
                return False
    return True

def read_and_warp(grid_file, xmin, ymin, xmax, ymax, req_height, req_width):
    """
    Read a geospatial raster and warp it to a target geographic extent and resolution.

    Processes raw IMERG GeoTIFF files by:
    1. Setting correct global georeference (-180 to 180, -90 to 90)
    2. Clipping to the region of interest
    3. Resampling to the target grid dimensions

    Args:
        grid_file (str): Path to the input GeoTIFF file.
        xmin (float): Western boundary longitude (decimal degrees).
        ymin (float): Southern boundary latitude (decimal degrees).
        xmax (float): Eastern boundary longitude (decimal degrees).
        ymax (float): Northern boundary latitude (decimal degrees).
        req_height (int): Desired output height in pixels.
        req_width (int): Desired output width in pixels.

    Returns:
        tuple: (data_array, width, height, geotransform, projection)
            - data_array: numpy array of precipitation values
            - width: output width in pixels
            - height: output height in pixels
            - geotransform: GDAL geotransform tuple
            - projection: WKT projection string

    Raises:
        RuntimeError: If the input file cannot be opened.

    Note:
        Creates a temporary file 'temp.tif' that should be cleaned up by caller.
    """
    # Open input raster
    raw_grid = gdal.Open(grid_file, GA_ReadOnly)
    if raw_grid is None:
        raise RuntimeError(f"Failed to open {grid_file}")

    # Set correct global extent and no-data value for IMERG data
    pre_ds = gdal.Translate('temp.tif', raw_grid,
                            options="-co COMPRESS=Deflate -a_nodata 29999 -a_ullr -180.0 90.0 180.0 -90.0")

    # Calculate pixel sizes for target dimensions
    pixel_size_x = (xmax - xmin) / req_width
    pixel_size_y = (ymax - ymin) / req_height

    # Warp to target extent and resolution using in-memory VRT
    ds = gdal.Warp('', pre_ds, srcNodata=29999, srcSRS='EPSG:4326', dstSRS='EPSG:4326',
                   dstNodata='-9999', format='VRT', xRes=pixel_size_x, yRes=-pixel_size_y,
                   outputBounds=(xmin, ymin, xmax, ymax))

    # Return data and metadata
    return ds.ReadAsArray(), ds.GetRasterBand(1).XSize, ds.GetRasterBand(1).YSize, \
           ds.GetGeoTransform(), ds.GetProjection()

def write_grid(grid_out_name, data_out, nx, ny, gt, proj):
    """
    Write processed precipitation data to a compressed GeoTIFF file.

    Creates a single-band Float32 GeoTIFF with DEFLATE compression,
    including proper georeferencing and no-data value.

    Args:
        grid_out_name (str): Output file path for the GeoTIFF.
        data_out (numpy.ndarray): 2D array of precipitation values (mm/hr).
        nx (int): Output width in pixels.
        ny (int): Output height in pixels.
        gt (tuple): GDAL GeoTransform tuple for spatial reference.
        proj (str): WKT projection string.

    Returns:
        None

    Side Effects:
        Creates a new GeoTIFF file at grid_out_name with:
        - DEFLATE compression
        - Float32 data type
        - No-data value of -9999.0
    """
    # Create output GeoTIFF with compression
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(grid_out_name, nx, ny, 1, gdal.GDT_Float32, ['COMPRESS=DEFLATE'])

    # Set geospatial metadata
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)

    # Reshape and write data
    data_out.shape = (-1, nx)
    dst_ds.GetRasterBand(1).WriteArray(data_out, 0, 0)
    dst_ds.GetRasterBand(1).SetNoDataValue(-9999.0)

    # Close file (flush to disk)
    dst_ds = None

def download_single(date, output_folder):
    """
    Download and process a single missing IMERG file for a specific timestamp.

    Handles the complete workflow for one time step:
    1. Constructs the IMERG filename from the target timestamp
    2. Downloads the raw file from NASA PPS
    3. Processes (warps and scales) the data
    4. Saves to the output folder
    5. Cleans up temporary files

    Note on timestamp handling:
        IMERG files are named by their START time, but we store them by their
        END time. So for an output file named imerg.202106011230.tif (ending at 12:30),
        we need to download the file that starts at 12:00 (30 minutes earlier).

    Args:
        date (datetime.datetime): Target timestamp for the output file.
            This is the END time of the 30-minute accumulation period.
        output_folder (str): Directory to save the processed GeoTIFF.

    Returns:
        bool: True if download and processing succeeded, False otherwise.

    Side Effects:
        - Downloads file from NASA PPS server
        - Creates processed GeoTIFF in output_folder
        - Creates and removes temporary files
        - Prints error messages on failure

    Note:
        Uses global configuration variables: SERVER_PATH, EMAIL, XMIN, YMIN,
        XMAX, YMAX, OUTPUT_HEIGHT, OUTPUT_WIDTH.
    """
    # IMERG filename components
    file_prefix = '3B-HHR-E.MS.MRG.3IMERG.'
    file_suffix = '.V07B.30min.tif'

    # Calculate the download timestamp (30 minutes before output timestamp)
    # IMERG files are named by START time; our output files use END time
    download_date = date - timedelta(minutes=30)

    # Build IMERG filename components
    initial_time_stmp = download_date.strftime('%Y%m%d-S%H%M%S')
    final_time_stmp = (download_date + timedelta(minutes=29)).strftime('E%H%M59')
    folder = download_date.strftime('%Y/%m/')  # Server folder structure
    total_minutes = download_date.hour * 60 + download_date.minute
    date_stamp = f"{initial_time_stmp}-{final_time_stmp}.{total_minutes:04}"

    # Construct file paths
    remote_filename = folder + file_prefix + date_stamp + file_suffix
    local_filename = file_prefix + date_stamp + file_suffix
    grid_out_name = os.path.join(output_folder, f"imerg.{date.strftime('%Y%m%d%H%M')}.tif")

    # Attempt to download the file
    get_file(remote_filename, SERVER_PATH, EMAIL)

    # Process if download was successful
    if os.path.isfile(local_filename):
        try:
            # Read, warp to target region and resolution
            new_grid, nx, ny, gt, proj = read_and_warp(
                local_filename, XMIN, YMIN, XMAX, YMAX, OUTPUT_HEIGHT, OUTPUT_WIDTH
            )
            # Scale from 0.1 mm/hr to mm/hr
            new_grid = new_grid * 0.1
            # Write processed output
            write_grid(grid_out_name, new_grid, nx, ny, gt, proj)
            return True
        except Exception as e:
            print(f"  Error: {e}")
            return False
        finally:
            # Always cleanup temporary files
            if os.path.isfile(local_filename):
                os.remove(local_filename)
            if os.path.isfile('temp.tif'):
                os.remove('temp.tif')
    return False

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
# This block executes when the script is run directly (not imported as a module).
# It scans for missing files, displays them, and optionally downloads them.

if __name__ == "__main__":
    # Print header and configuration summary
    print("=" * 60)
    print("IMERG Missing File Checker & Downloader")
    print("=" * 60)

    # Phase 1: Scan for missing files
    # This checks every 30-minute interval in the date range
    print(f"\nChecking for missing files from {START_DATE} to {END_DATE}...")
    missing = find_missing_files(START_DATE, END_DATE, IMERG_DATA_FOLDER)

    print(f"\nFound {len(missing)} missing files")

    # Exit early if no files are missing
    if not missing:
        print("All files present!")
        exit(0)

    # Phase 2: Display all missing file timestamps
    # This helps the user understand the scope of the gaps
    print("\nMissing files:")
    for i, d in enumerate(missing):
        print(f"  {i+1}. {d.strftime('%Y-%m-%d %H:%M')} -> imerg.{d.strftime('%Y%m%d%H%M')}.tif")

    print(f"\nTotal: {len(missing)} missing files")

    # Phase 3: Check dry run mode
    # In dry run mode, we only report missing files without downloading
    if DRY_RUN:
        print("\n" + "=" * 60)
        print("DRY RUN MODE - No files downloaded")
        print("To download, set DRY_RUN = False in the script")
        print("=" * 60)
        exit(0)

    # Phase 4: Download missing files
    # Only executed when DRY_RUN = False
    print(f"\nDownloading {len(missing)} missing files...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Track success/failure counts for summary
    success = 0
    failed = 0

    # Download each missing file with progress indicator
    for i, date in enumerate(missing):
        print(f"[{i+1}/{len(missing)}] Downloading {date.strftime('%Y-%m-%d %H:%M')}", end=" ")
        if download_single(date, OUTPUT_FOLDER):
            print("OK")
            success += 1
        else:
            print("FAILED")
            failed += 1

    # Print final summary
    print(f"\nDone! Success: {success}, Failed: {failed}")
