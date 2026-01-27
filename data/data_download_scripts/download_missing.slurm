#!/bin/bash
# =============================================================================
# SLURM Job Script: IMERG Missing File Downloader
# =============================================================================
#
# Description:
#   This SLURM batch script identifies and downloads missing NASA GPM IMERG
#   precipitation files for an existing data directory. It wraps the Python
#   script download_missing_imerg.py which scans for gaps in the time series.
#
# Purpose:
#   - Gap-filling after interrupted downloads
#   - Data validation and completeness checks
#   - Recovery from server outages or network issues
#
# Workflow:
#   1. First run (DRY_RUN=True in Python script):
#      - Scans the data directory for missing files
#      - Reports all gaps without downloading
#      - Use this to assess the scope of missing data
#
#   2. Second run (DRY_RUN=False in Python script):
#      - Downloads and processes all missing files
#      - Provides progress tracking and success/failure counts
#
# Usage:
#   1. Submit job: sbatch download_missing.slurm
#   2. Review output: cat /home1/ppatel2025/ldcast/data/logs/imerg_missing_<jobid>.out
#   3. If DRY_RUN=True, edit download_missing_imerg.py and set DRY_RUN=False
#   4. Re-submit to download: sbatch download_missing.slurm
#
# Configuration:
#   All configuration (date range, paths, credentials) is set in the Python
#   script download_missing_imerg.py, NOT in this SLURM script.
#
# Output:
#   - Job logs: /home1/ppatel2025/ldcast/data/logs/imerg_missing_<jobid>.out
#   - Error logs: /home1/ppatel2025/ldcast/data/logs/imerg_missing_<jobid>.err
#   - Downloaded files: Specified in download_missing_imerg.py
#
# Prerequisites:
#   - Existing IMERG data directory (from download_imerg_data.py)
#   - NASA Earthdata account configured in download_missing_imerg.py
#   - Conda environment 'tito_env' with GDAL and dependencies
#
# =============================================================================

# -----------------------------------------------------------------------------
# SLURM Resource Configuration
# -----------------------------------------------------------------------------
# Resource requests for the HPC scheduler.
# Note: There are duplicate --ntasks and --cpus-per-task directives below;
# the later values (1 and 1) will be used by SLURM.

#SBATCH --job-name=imerg_missing            # Job name shown in squeue
#SBATCH --nodes=1                           # Number of compute nodes
#SBATCH --ntasks=1                          # Number of tasks (serial job)
#SBATCH --cpus-per-task=16                  # CPUs (note: overridden below)
#SBATCH --output=/home1/ppatel2025/ldcast/data/logs/imerg_missing_%j.out  # Stdout log
#SBATCH --error=/home1/ppatel2025/ldcast/data/logs/imerg_missing_%j.err   # Stderr log
#SBATCH --ntasks=1                          # Number of tasks (duplicate)
#SBATCH --cpus-per-task=1                   # CPUs per task (this value used)
#SBATCH --mem=32G                           # Memory allocation

# =============================================================================
# EXECUTION
# =============================================================================

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
# Initialize conda and activate the environment with required dependencies

source /home1/ppatel2025/miniconda3/etc/profile.d/conda.sh
conda activate tito_env

# -----------------------------------------------------------------------------
# Change to Script Directory
# -----------------------------------------------------------------------------
# Required for the Python script to find files and create temp files

cd /home1/ppatel2025/ldcast/data/data_download_scripts

# -----------------------------------------------------------------------------
# Run the Missing File Checker/Downloader
# -----------------------------------------------------------------------------
# IMPORTANT: The download behavior is controlled by DRY_RUN in the Python script
#
# DRY_RUN = True (default):
#   - Only LISTS missing files without downloading
#   - Safe for initial assessment of data gaps
#
# DRY_RUN = False:
#   - Actually DOWNLOADS and processes missing files
#   - Edit download_missing_imerg.py to enable this mode
#
# All other configuration (date range, paths, credentials) is set in
# download_missing_imerg.py, not in this SLURM script.

echo "============================================================"
echo "IMERG Missing File Checker/Downloader"
echo "============================================================"
echo "NOTE: Check DRY_RUN setting in download_missing_imerg.py"
echo "      DRY_RUN=True  -> Only lists missing files"
echo "      DRY_RUN=False -> Downloads missing files"
echo "============================================================"

python download_missing_imerg.py

echo "============================================================"
echo "Job completed."
echo "============================================================"
