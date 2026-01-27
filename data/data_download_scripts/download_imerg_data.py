"""
Simple IMERG Data Downloader

This script downloads NASA GPM IMERG (Integrated Multi-satellitE Retrievals for GPM)
precipitation data for a specified geographic region and time period. The downloaded
data is processed, warped to a target grid, and saved as GeoTIFF files.

IMERG provides global precipitation estimates at 0.1° x 0.1° resolution with
30-minute temporal resolution. This script specifically downloads the Early Run
product (3B-HHR-E), which is available with ~4 hour latency.

Features:
    - Downloads IMERG Early Run half-hourly precipitation data
    - Automatically clips to user-defined geographic bounds
    - Resamples to user-specified output grid dimensions
    - Handles authentication with NASA PPS (Precipitation Processing System)
    - Skips already-downloaded files to support resumable downloads
    - Cleans up temporary files automatically

Requirements:
    - NASA Earthdata account (register at https://urs.earthdata.nasa.gov/)
    - GDAL Python bindings (osgeo)
    - curl command-line tool

Usage:
    Set environment variables or edit defaults below, then run the script.
    Can be called from SLURM script with exported variables.

Environment Variables:
    XMIN, XMAX, YMIN, YMAX: Geographic bounds (longitude/latitude)
    OUTPUT_HEIGHT, OUTPUT_WIDTH: Output grid dimensions in pixels
    START_DATE, END_DATE: Time period in format 'YYYY-MM-DD-HH-MM'
    OUTPUT_FOLDER: Directory to save processed GeoTIFF files
    EMAIL: NASA Earthdata registered email for authentication
    SERVER_PATH: NASA PPS server URL

Example:
    $ export START_DATE="2021-06-01-00-00"
    $ export END_DATE="2021-06-02-00-00"
    $ python download_imerg_data.py

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
# CONFIGURATION - Read from environment variables or use defaults
# =============================================================================
# All configuration parameters can be overridden via environment variables,
# making this script suitable for batch processing via SLURM or other job
# schedulers without modifying the source code.

# Geographic bounding box for the region of interest
# Coordinates are in decimal degrees (WGS84 / EPSG:4326)
# Default values cover Burkina Faso in West Africa
XMIN = float(os.environ.get('XMIN', '-5.5'))   # Western boundary (longitude)
XMAX = float(os.environ.get('XMAX', '2.5'))    # Eastern boundary (longitude)
YMIN = float(os.environ.get('YMIN', '9.0'))    # Southern boundary (latitude)
YMAX = float(os.environ.get('YMAX', '17.0'))   # Northern boundary (latitude)

# Output grid dimensions in pixels
# The downloaded IMERG data will be resampled to this resolution
# 64x64 is suitable for machine learning applications
OUTPUT_HEIGHT = int(os.environ.get('OUTPUT_HEIGHT', '64'))  # Number of rows
OUTPUT_WIDTH = int(os.environ.get('OUTPUT_WIDTH', '64'))    # Number of columns

# Time period - parse from environment or use defaults
def parse_date(date_str, default):
    """
    Parse a date string into a datetime object.

    Converts a hyphen-separated date string into a Python datetime object.
    This function is used to parse date strings from environment variables
    for specifying the download time period.

    Args:
        date_str (str or None): Date string in format 'YYYY-MM-DD-HH-MM' where:
            - YYYY: 4-digit year
            - MM: 2-digit month (01-12)
            - DD: 2-digit day (01-31)
            - HH: 2-digit hour (00-23)
            - MM: 2-digit minute (00-59)
            Example: '2021-06-15-12-30' for June 15, 2021 at 12:30 PM
        default (datetime.datetime): Default datetime to return if date_str
            is None or empty.

    Returns:
        datetime.datetime: Parsed datetime object with seconds set to 0,
            or the default value if date_str is falsy.

    Raises:
        IndexError: If date_str has fewer than 5 hyphen-separated parts.
        ValueError: If any part cannot be converted to an integer.

    Example:
        >>> parse_date('2021-06-01-00-00', datetime.datetime(2020, 1, 1))
        datetime.datetime(2021, 6, 1, 0, 0, 0)
        >>> parse_date(None, datetime.datetime(2020, 1, 1))
        datetime.datetime(2020, 1, 1, 0, 0, 0)
    """
    if date_str:
        parts = date_str.split('-')
        return datetime.datetime(int(parts[0]), int(parts[1]), int(parts[2]),
                                  int(parts[3]), int(parts[4]), 0)
    return default


# Time period for data download
# Dates should be provided in 'YYYY-MM-DD-HH-MM' format via environment variables
# The download will include all 30-minute intervals from START_DATE to END_DATE (inclusive)
START_DATE = parse_date(os.environ.get('START_DATE'),
                        datetime.datetime(2014, 6, 1, 0, 0, 0))  # Default: June 1, 2014 00:00
END_DATE = parse_date(os.environ.get('END_DATE'),
                      datetime.datetime(2014, 6, 3, 0, 0, 0))    # Default: June 3, 2014 00:00

# Output directory for processed GeoTIFF files
# Files will be named: imerg.YYYYMMDDHHMM.tif
OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', '/home1/ppatel2025/ldcast/imerg_data/')

# NASA PPS authentication credentials
# Register at https://urs.earthdata.nasa.gov/ to get your credentials
# NASA PPS uses the email address as both username and password
EMAIL = os.environ.get('EMAIL', 'aaravamudan2014@my.fit.edu')

# NASA PPS IMERG data server URL
# 'early/' = Early Run (~4hr latency, lower accuracy)
# 'late/'  = Late Run (~14hr latency, better accuracy)
# 'final/' = Final Run (~3.5 months latency, research quality)
SERVER_PATH = os.environ.get('SERVER_PATH',
                             'https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/')

# =============================================================================
# FUNCTIONS
# =============================================================================

def get_file(filename, server, email):
    """
    Download a file from the NASA PPS (Precipitation Processing System) server.

    Uses curl to download IMERG data files with HTTP Basic Authentication.
    NASA PPS uses the same email address for both username and password
    in their authentication scheme.

    The function also validates the downloaded file to ensure it's not an
    HTML error page (which can occur when the file doesn't exist on the
    server or authentication fails).

    Args:
        filename (str): Relative path to the file on the server, including
            any subdirectories (e.g., '2021/06/3B-HHR-E.MS.MRG.3IMERG...tif').
        server (str): Base URL of the NASA PPS server
            (e.g., 'https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/').
        email (str): NASA Earthdata registered email address used for
            authentication (same value used for both username and password).

    Returns:
        bool: True if the file was successfully downloaded and appears to be
            a valid binary file (not an HTML error page). False if the download
            failed or the file is an HTML error page.

    Side Effects:
        - Creates a file in the current working directory with the basename
          of the filename parameter.
        - If the downloaded file is detected as HTML, it is automatically deleted.

    Note:
        The -s flag in curl suppresses progress output.
        The -O flag saves the file with its remote filename.
        The -u flag provides authentication credentials.
    """
    # Construct the full URL by joining server base and relative filename
    url = server + '/' + filename
    local_name = os.path.basename(filename)

    # Build curl command with silent mode and HTTP Basic Auth
    # NASA PPS uses email as both username and password
    cmd = f'curl -sO -u {email}:{email} {url}'
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()

    # Validate the downloaded file
    # NASA PPS returns HTML error pages instead of 404 errors when files are missing
    # We detect this by checking if the file starts with HTML markers
    if os.path.isfile(local_name):
        with open(local_name, 'rb') as f:
            header = f.read(100)
            # Check for common HTML document markers in the file header
            if b'<!DOCTYPE' in header or b'<html' in header.lower():
                os.remove(local_name)  # Remove the invalid HTML file
                return False
    return True

def read_and_warp(grid_file, xmin, ymin, xmax, ymax, req_height, req_width):
    """
    Read a geospatial raster file and warp it to a specified geographic extent and resolution.

    This function performs several geospatial operations on IMERG precipitation data:
    1. Opens the raw IMERG GeoTIFF file
    2. Corrects the georeference to global extent (-180 to 180 longitude, -90 to 90 latitude)
    3. Reprojects and resamples the data to match the target region and grid dimensions

    IMERG data comes as global coverage files. This function extracts only the region
    of interest and resamples it to the desired output grid size for further processing.

    Args:
        grid_file (str): Path to the input GeoTIFF file to be processed.
        xmin (float): Western boundary longitude in decimal degrees (e.g., -5.5).
        ymin (float): Southern boundary latitude in decimal degrees (e.g., 9.0).
        xmax (float): Eastern boundary longitude in decimal degrees (e.g., 2.5).
        ymax (float): Northern boundary latitude in decimal degrees (e.g., 17.0).
        req_height (int): Desired output grid height in pixels (number of rows).
        req_width (int): Desired output grid width in pixels (number of columns).

    Returns:
        tuple: A 5-element tuple containing:
            - warped_grid (numpy.ndarray): 2D array of precipitation values clipped
              and resampled to the target region and resolution.
            - new_nx (int): Width of the output grid in pixels.
            - new_ny (int): Height of the output grid in pixels.
            - new_gt (tuple): GDAL GeoTransform tuple (origin_x, pixel_width, 0,
              origin_y, 0, pixel_height) for the output grid.
            - new_proj (str): WKT projection string (EPSG:4326 / WGS84).

    Raises:
        RuntimeError: If the input file cannot be opened (corrupted or invalid format).

    Note:
        - IMERG raw files use 29999 as the no-data value; this is converted to -9999
          in the output for consistency.
        - The function creates a temporary file 'temp.tif' which should be cleaned
          up by the caller.
        - Uses VRT (Virtual Raster) format for memory-efficient warping.
    """
    # Open the input raster file in read-only mode
    raw_grid = gdal.Open(grid_file, GA_ReadOnly)

    # Validate that the file was opened successfully
    if raw_grid is None:
        raise RuntimeError(f"Failed to open {grid_file} - file may be corrupted or not a valid raster")

    # IMERG GeoTIFFs may not have correct georeference metadata
    # Use gdal.Translate to set the correct global extent (-180 to 180, -90 to 90)
    # and define the no-data value used in IMERG files (29999)
    pre_ds = gdal.Translate('temp.tif', raw_grid, options="-co COMPRESS=Deflate -a_nodata 29999 -a_ullr -180.0 90.0 180.0 -90.0")

    gt = pre_ds.GetGeoTransform()
    no_data = 29999  # IMERG uses 29999 as no-data/missing value

    # Calculate the pixel size needed to achieve the requested grid dimensions
    # Pixel size = total extent / number of pixels
    pixel_size_x = (xmax - xmin) / req_width
    pixel_size_y = (ymax - ymin) / req_height

    # Warp (reproject and resample) the data to the target extent and resolution
    # Using VRT format keeps the operation in memory without writing to disk
    # Note: yRes is negative because raster rows go from top to bottom
    ds = gdal.Warp('', pre_ds, srcNodata=no_data, srcSRS='EPSG:4326', dstSRS='EPSG:4326',
                   dstNodata='-9999', format='VRT', xRes=pixel_size_x, yRes=-pixel_size_y,
                   outputBounds=(xmin, ymin, xmax, ymax))

    # Extract the warped data and metadata
    warped_grid = ds.ReadAsArray()  # Get the pixel values as a numpy array
    new_gt = ds.GetGeoTransform()   # Get the new geotransform
    new_proj = ds.GetProjection()   # Get the projection (WKT format)
    new_nx = ds.GetRasterBand(1).XSize  # Output width in pixels
    new_ny = ds.GetRasterBand(1).YSize  # Output height in pixels

    return warped_grid, new_nx, new_ny, new_gt, new_proj

def write_grid(grid_out_name, data_out, nx, ny, gt, proj):
    """
    Write processed precipitation data to a compressed GeoTIFF file.

    Creates a single-band GeoTIFF file with the processed precipitation data,
    including proper georeferencing information and compression for efficient
    storage.

    Args:
        grid_out_name (str): Full path for the output GeoTIFF file
            (e.g., '/path/to/imerg.202106011200.tif').
        data_out (numpy.ndarray): 2D array of precipitation values (in mm/hr
            after scaling). Can be any shape that can be reshaped to (ny, nx).
        nx (int): Width of the output grid in pixels (number of columns).
        ny (int): Height of the output grid in pixels (number of rows).
        gt (tuple): GDAL GeoTransform tuple defining the spatial reference:
            (top_left_x, pixel_width, rotation_x, top_left_y, rotation_y, pixel_height).
            For north-up images, rotations are 0 and pixel_height is negative.
        proj (str): Projection definition in WKT (Well-Known Text) format,
            typically EPSG:4326 for geographic coordinates.

    Returns:
        None

    Side Effects:
        - Creates a new GeoTIFF file at grid_out_name.
        - Uses DEFLATE compression to reduce file size.
        - Sets no-data value to -9999.0 for missing data pixels.

    Note:
        - Output uses Float32 data type to preserve precipitation precision.
        - The data_out array is reshaped in-place to match (ny, nx) dimensions.
        - Setting dst_ds to None triggers GDAL to flush and close the file.
    """
    # Get the GeoTIFF driver for creating the output file
    driver = gdal.GetDriverByName('GTiff')

    # Create the output file with:
    # - Single band (1)
    # - 32-bit floating point data type for precipitation values
    # - DEFLATE compression to reduce file size
    dst_ds = driver.Create(grid_out_name, nx, ny, 1, gdal.GDT_Float32, ['COMPRESS=DEFLATE'])

    # Set the geospatial metadata
    dst_ds.SetGeoTransform(gt)   # Define pixel coordinates to geographic coordinates mapping
    dst_ds.SetProjection(proj)  # Set the coordinate reference system

    # Reshape the data array to match the expected 2D grid dimensions
    # This handles cases where the array might be flattened or have wrong shape
    data_out.shape = (-1, nx)

    # Write the precipitation data to the first (and only) band
    # WriteArray(data, xoff, yoff) - starting at pixel (0, 0)
    dst_ds.GetRasterBand(1).WriteArray(data_out, 0, 0)

    # Set the no-data value so GIS software knows which pixels are missing
    dst_ds.GetRasterBand(1).SetNoDataValue(-9999.0)

    # Close the dataset by dereferencing - this flushes data to disk
    dst_ds = None

def download_imerg(output_folder, start_date, end_date, xmin, ymin, xmax, ymax,
                   req_height, req_width, server, email):
    """
    Download and process IMERG precipitation data for a specified region and time period.

    This is the main orchestration function that iterates through each 30-minute
    time step, downloads the corresponding IMERG file from NASA PPS, processes it
    to extract the region of interest, and saves the result as a GeoTIFF.

    The function implements:
    - Resumable downloads (skips files that already exist)
    - Automatic cleanup of temporary files
    - Error handling for corrupted or missing files
    - Progress reporting to stdout

    IMERG Filename Convention:
        3B-HHR-E.MS.MRG.3IMERG.YYYYMMDD-SHHMMSS-EHHMMSS.MMMM.V07B.30min.tif
        Where:
        - 3B-HHR-E: Product code (Half-Hourly Early Run)
        - YYYYMMDD: Date
        - SHHMMSS: Start time (S = Start)
        - EHHMMSS: End time (E = End)
        - MMMM: Minutes since midnight (0000-1410)
        - V07B: Algorithm version

    Args:
        output_folder (str): Directory path where processed GeoTIFF files will
            be saved. Created if it doesn't exist.
        start_date (datetime.datetime): Start of the download period (inclusive).
        end_date (datetime.datetime): End of the download period (inclusive).
        xmin (float): Western boundary longitude in decimal degrees.
        ymin (float): Southern boundary latitude in decimal degrees.
        xmax (float): Eastern boundary longitude in decimal degrees.
        ymax (float): Northern boundary latitude in decimal degrees.
        req_height (int): Desired output grid height in pixels.
        req_width (int): Desired output grid width in pixels.
        server (str): Base URL of the NASA PPS IMERG data server.
        email (str): NASA Earthdata registered email for authentication.

    Returns:
        None

    Side Effects:
        - Creates the output_folder directory if it doesn't exist.
        - Downloads files from NASA PPS server.
        - Creates GeoTIFF files in output_folder with naming pattern:
          'imerg.YYYYMMDDHHMM.tif'
        - Prints progress messages to stdout.
        - Creates and deletes temporary files during processing.

    Note:
        - Precipitation values are scaled by 0.1 to convert from IMERG's
          native units (0.1 mm/hr) to mm/hr.
        - Each 30-minute file covers the period [timestamp-30min, timestamp).
        - The output filename uses the END time of the 30-minute period.
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # IMERG filename components
    # 3B-HHR-E = Half-Hourly Early Run product
    file_prefix = '3B-HHR-E.MS.MRG.3IMERG.'
    file_suffix = '.V07B.30min.tif'  # Version 07B, 30-minute accumulation
    delta_time = timedelta(minutes=30)  # IMERG temporal resolution

    # Initialize time loop
    current_date = start_date
    # Add 30 minutes to end_date to include the last time step
    final_date = end_date + timedelta(minutes=30)

    # Iterate through each 30-minute time step
    while current_date < final_date:
        # Build the IMERG filename components
        # Start timestamp: YYYYMMDD-SHHMMSS
        initial_time_stmp = current_date.strftime('%Y%m%d-S%H%M%S')

        # End timestamp: EHHMMSS (29 minutes after start, ending at :59 seconds)
        final_time = current_date + timedelta(minutes=29)
        final_time_stmp = final_time.strftime('E%H%M59')

        # Output file timestamp (end of 30-minute period)
        final_time_gridout = current_date + timedelta(minutes=30)

        # Server folder structure: YYYY/MM/
        folder = current_date.strftime('%Y/%m/')

        # Calculate minutes since midnight for the filename
        # IMERG uses this as a unique identifier within each day
        total_minutes = current_date.hour * 60 + current_date.minute
        date_stamp = f"{initial_time_stmp}-{final_time_stmp}.{total_minutes:04}"

        # Construct full filename paths
        filename = folder + file_prefix + date_stamp + file_suffix
        local_filename = file_prefix + date_stamp + file_suffix
        grid_out_name = output_folder + 'imerg.' + final_time_gridout.strftime('%Y%m%d%H%M') + '.tif'

        print(f'Downloading {final_time_gridout.strftime("%Y-%m-%d %H:%M")}')

        # Skip if output file already exists (resumable downloads)
        if not os.path.isfile(grid_out_name):
            # Attempt to download the file from NASA PPS
            get_file(filename, server, email)

            # Check if download was successful
            if os.path.isfile(local_filename):
                try:
                    # Process the downloaded file:
                    # 1. Read the raw data
                    # 2. Warp to target region and resolution
                    new_grid, nx, ny, gt, proj = read_and_warp(
                        local_filename, xmin, ymin, xmax, ymax, req_height, req_width
                    )

                    # Scale precipitation values from 0.1 mm/hr to mm/hr
                    # IMERG stores values as integers with 0.1 mm/hr precision
                    new_grid = new_grid * 0.1

                    # Write the processed data to output GeoTIFF
                    write_grid(grid_out_name, new_grid, nx, ny, gt, proj)
                except RuntimeError as e:
                    print(f'  Warning: {e}')
                finally:
                    # Always cleanup temporary files, even if processing failed
                    if os.path.isfile(local_filename):
                        os.remove(local_filename)
                    if os.path.isfile('temp.tif'):
                        os.remove('temp.tif')
            else:
                # Download failed (file not found on server or auth error)
                print(f'  Warning: Failed to download {local_filename}')
        else:
            # Output already exists, skip to save time
            print(f'  File already exists, skipping')

        # Move to next 30-minute time step
        current_date += delta_time

    print(f'\nDownload complete! Files saved to: {output_folder}')

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
# This block executes when the script is run directly (not imported as a module).
# It prints a summary of the configuration and starts the download process.

if __name__ == "__main__":
    # Print configuration summary for user verification
    print("=" * 60)
    print("IMERG Data Downloader")
    print("=" * 60)
    print(f"Region: [{XMIN}, {XMAX}] x [{YMIN}, {YMAX}]")  # Lon x Lat bounds
    print(f"Grid size: {OUTPUT_WIDTH} x {OUTPUT_HEIGHT}")  # Output dimensions
    print(f"Period: {START_DATE} to {END_DATE}")           # Time range
    print(f"Output: {OUTPUT_FOLDER}")                      # Save location
    print("=" * 60)

    # Execute the main download and processing function
    # All parameters are passed explicitly for clarity and testability
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