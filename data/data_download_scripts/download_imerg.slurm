#!/bin/bash
# =============================================================================
# SLURM Job Script: IMERG Data Downloader
# =============================================================================
#
# Description:
#   This SLURM batch script downloads NASA GPM IMERG precipitation data for
#   a specified geographic region and time period. It wraps the Python script
#   download_imerg_data.py and passes configuration via environment variables.
#
# IMERG Product:
#   Downloads Early Run (3B-HHR-E) half-hourly precipitation estimates at
#   0.1° resolution, then clips and resamples to the specified region and grid.
#
# Usage:
#   1. Edit the CONFIGURATION section below with your desired parameters
#   2. Submit the job: sbatch download_imerg.slurm
#   3. Monitor progress: tail -f /home1/ppatel2025/ldcast/data/logs/imerg_download_<jobid>.out
#
# Output:
#   - Processed GeoTIFF files: <OUTPUT_FOLDER>/imerg.YYYYMMDDHHMM.tif
#   - Job logs: /home1/ppatel2025/ldcast/data/logs/imerg_download_<jobid>.out
#   - Error logs: /home1/ppatel2025/ldcast/data/logs/imerg_download_<jobid>.err
#
# Prerequisites:
#   - NASA Earthdata account (email registered with PPS)
#   - Conda environment 'tito_env' with GDAL and dependencies
#   - Network access to NASA PPS server
#
# =============================================================================

# -----------------------------------------------------------------------------
# SLURM Resource Configuration
# -----------------------------------------------------------------------------
# These directives tell SLURM how to schedule and resource this job.
# Adjust based on your cluster's available partitions and your needs.

#SBATCH --job-name=imerg_download           # Job name shown in squeue
#SBATCH --partition=h200                    # Partition/queue to submit to
#SBATCH --nodes=1                           # Number of compute nodes (1 is sufficient)
#SBATCH --ntasks=1                          # Number of tasks (1 for serial job)
#SBATCH --cpus-per-task=4                   # CPUs per task (for GDAL parallelism)
#SBATCH --mem=32G                           # Memory allocation
#SBATCH --output=/home1/ppatel2025/ldcast/data/logs/imerg_download_%j.out  # Stdout log (%j = job ID)
#SBATCH --error=/home1/ppatel2025/ldcast/data/logs/imerg_download_%j.err   # Stderr log (%j = job ID)

# =============================================================================
# CONFIGURATION - Edit these variables as needed
# =============================================================================
# These environment variables are read by download_imerg_data.py
# All parameters can be customized for your specific use case.

# -----------------------------------------------------------------------------
# Geographic Bounding Box
# -----------------------------------------------------------------------------
# Coordinates in decimal degrees (WGS84 / EPSG:4326)
# Default values define a bounding box over Burkina Faso, West Africa

export XMIN="-5.5"      # Western boundary (longitude, degrees East)
export XMAX="2.5"       # Eastern boundary (longitude, degrees East)
export YMIN="9.0"       # Southern boundary (latitude, degrees North)
export YMAX="17.0"      # Northern boundary (latitude, degrees North)

# -----------------------------------------------------------------------------
# Output Grid Dimensions
# -----------------------------------------------------------------------------
# The downloaded global IMERG data will be clipped and resampled to this grid size.
# 64x64 is a common choice for machine learning applications.

export OUTPUT_HEIGHT="64"   # Number of rows (pixels in Y direction)
export OUTPUT_WIDTH="64"    # Number of columns (pixels in X direction)

# -----------------------------------------------------------------------------
# Time Period
# -----------------------------------------------------------------------------
# Date format: YYYY-MM-DD-HH-MM
# Downloads all 30-minute intervals from START_DATE to END_DATE (inclusive)
# Example: 2 days of data = 96 files (48 per day × 2 days)

export START_DATE="2014-06-01-00-00"    # Start datetime
export END_DATE="2014-06-03-00-00"      # End datetime

# -----------------------------------------------------------------------------
# Output Directory
# -----------------------------------------------------------------------------
# Processed GeoTIFF files will be saved here with naming pattern:
# imerg.YYYYMMDDHHMM.tif (e.g., imerg.201406011230.tif)

export OUTPUT_FOLDER="/home1/ppatel2025/ldcast/data/imerg_data/"

# -----------------------------------------------------------------------------
# NASA PPS Authentication
# -----------------------------------------------------------------------------
# Your NASA Earthdata email address (register at https://urs.earthdata.nasa.gov/)
# NASA PPS uses the same email for both username and password

export EMAIL="aaravamudan2014@my.fit.edu"

# -----------------------------------------------------------------------------
# IMERG Data Server
# -----------------------------------------------------------------------------
# NASA PPS server URL. Available products:
#   - early/  : Early Run (~4 hour latency, near-real-time)
#   - late/   : Late Run (~14 hour latency, better accuracy)
#   - final/  : Final Run (~3.5 months latency, research quality)

export SERVER_PATH="https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/"

# =============================================================================
# EXECUTION
# =============================================================================
# This section sets up the environment and runs the download script.
# Generally, you shouldn't need to modify this section.

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
# Initialize conda and activate the environment with required dependencies
# (GDAL, numpy, etc.)

source /home1/ppatel2025/miniconda3/etc/profile.d/conda.sh
conda activate tito_env

# Change to the script directory (required for relative imports/temp files)
SCRIPT_ROOT="/home1/ppatel2025/ldcast/data/data_download_scripts"
cd $SCRIPT_ROOT

# -----------------------------------------------------------------------------
# Print Configuration Summary
# -----------------------------------------------------------------------------
# Log the configuration for debugging and verification

echo "============================================================"
echo "Starting IMERG download..."
echo "============================================================"
echo "Region: XMIN=$XMIN, XMAX=$XMAX, YMIN=$YMIN, YMAX=$YMAX"
echo "Grid size: ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT}"
echo "Period: $START_DATE to $END_DATE"
echo "Output: $OUTPUT_FOLDER"
echo "============================================================"

# -----------------------------------------------------------------------------
# Run the Download Script
# -----------------------------------------------------------------------------
# Execute the Python script which reads configuration from environment variables

python download_imerg_data.py

# -----------------------------------------------------------------------------
# Completion
# -----------------------------------------------------------------------------
echo "============================================================"
echo "Job completed."
echo "============================================================"
