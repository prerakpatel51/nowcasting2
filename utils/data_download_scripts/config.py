"""
Configuration File for IMERG Data Download Scripts

This module provides centralized configuration for all IMERG data download scripts.
All paths are computed relative to the project root, ensuring the scripts work
correctly regardless of the current working directory.

Configuration Sections:
    - Path Configuration: Project root, data directories, log directories
    - Geographic Configuration: Bounding box coordinates
    - Grid Configuration: Output dimensions
    - Time Configuration: Default date range
    - NASA PPS Configuration: Server URL and authentication
    - HDF5 Configuration: Conversion settings for HDF5 output

Usage:
    from config import (
        DATA_DIR, LOG_DIR, IMERG_DATA_DIR,
        XMIN, XMAX, YMIN, YMAX,
        OUTPUT_HEIGHT, OUTPUT_WIDTH,
        EMAIL, SERVER_PATH,
        HDF5_OUTPUT_DIR, HDF5_FILENAME, HDF5_BATCH_SIZE
    )
"""

import os
import datetime

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# All paths are computed relative to the project root to ensure portability.
# The project root is determined by navigating up from this config file's location.

# Directory containing this config file (utils/data_download_scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root directory (two levels up from script directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Data directory (PROJECT_ROOT/data/)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# IMERG data storage directory
IMERG_DATA_DIR = os.path.join(DATA_DIR, 'imerg_data')

# Log directory for SLURM job outputs
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs', 'download_logs')

# =============================================================================
# GEOGRAPHIC CONFIGURATION
# =============================================================================
# Bounding box coordinates in decimal degrees (WGS84 / EPSG:4326)
# Default region: Burkina Faso, West Africa

XMIN = -5.5    # Western boundary (longitude)
XMAX = 2.5     # Eastern boundary (longitude)
YMIN = 9.0     # Southern boundary (latitude)
YMAX = 17.0    # Northern boundary (latitude)

# =============================================================================
# GRID CONFIGURATION
# =============================================================================
# Output grid dimensions in pixels
# The downloaded IMERG data will be resampled to this resolution

OUTPUT_HEIGHT = 64  # Number of rows (pixels in Y direction)
OUTPUT_WIDTH = 64   # Number of columns (pixels in X direction)

# =============================================================================
# TIME CONFIGURATION
# =============================================================================
# Default date range for downloads
# Format: datetime.datetime(year, month, day, hour, minute, second)

DEFAULT_START_DATE = datetime.datetime(2011, 4, 1, 0, 0, 0)
DEFAULT_END_DATE = datetime.datetime(2022, 12, 31, 0, 0, 0)

# For missing file checker (longer range)
MISSING_CHECK_START_DATE = datetime.datetime(2011, 4, 1, 0, 30, 0)
MISSING_CHECK_END_DATE = datetime.datetime(2022, 12, 31, 23, 30, 0)

# =============================================================================
# NASA PPS CONFIGURATION
# =============================================================================
# NASA Precipitation Processing System server and authentication settings
# Register at https://urs.earthdata.nasa.gov/ to get credentials

# Registered email address (used as both username and password)
EMAIL = 'aaravamudan2014@my.fit.edu'

# NASA PPS IMERG data server URL
# Available products:
#   - early/  : Early Run (~4 hour latency, near-real-time)
#   - late/   : Late Run (~14 hour latency, better accuracy)
#   - final/  : Final Run (~3.5 months latency, research quality)
SERVER_PATH = 'https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/'

# =============================================================================
# CONDA ENVIRONMENT CONFIGURATION
# =============================================================================
# Path to conda installation for SLURM scripts

CONDA_PATH = '/home1/ppatel2025/miniconda3/etc/profile.d/conda.sh'
CONDA_ENV = 'tito_env'

# =============================================================================
# DOWNLOAD SETTINGS
# =============================================================================
# Dry run mode for missing file downloader
# When True: Only lists missing files without downloading
# When False: Actually downloads and processes missing files

DRY_RUN = False

# =============================================================================
# HDF5 CONVERSION CONFIGURATION
# =============================================================================
# Settings for converting IMERG GeoTIFF files to HDF5 format

# Output directory for HDF5 file
HDF5_OUTPUT_DIR = DATA_DIR

# HDF5 output filename
HDF5_FILENAME = 'imerg_data.h5'

# Batch size for processing (number of files to load at once)
# Increase for faster processing if memory allows
# Decrease if running out of memory
HDF5_BATCH_SIZE = 1000

# Chunk size for HDF5 datasets (48 = 1 day of 30-min data)
# Affects read performance for dataloaders
HDF5_TEMPORAL_CHUNK_SIZE = 48

# Compression algorithm: 'lzf' (fast) or 'gzip' (smaller)
HDF5_COMPRESSION = 'lzf'

# NoData value for missing/invalid precipitation data
HDF5_NODATA_VALUE = -9999.0


def ensure_directories():
    """
    Create required directories if they don't exist.

    Creates the data directory, IMERG data directory, and log directory.
    Call this function at the start of download scripts to ensure
    all necessary directories are in place.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMERG_DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def get_output_folder():
    """
    Get the IMERG output folder path, creating it if necessary.

    Returns:
        str: Absolute path to the IMERG data directory.
    """
    ensure_directories()
    return IMERG_DATA_DIR
