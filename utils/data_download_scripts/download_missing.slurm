#!/bin/bash
# =============================================================================
# SLURM Job Script: IMERG Missing File Downloader
# =============================================================================
#
# Description:
#   Identifies and downloads missing NASA GPM IMERG precipitation files.
#   Wraps download_missing_imerg.py which reads configuration from config.py.
#
# Usage:
#   sbatch download_missing.slurm
#
# Configuration:
#   All parameters are set in config.py:
#   - DRY_RUN: Set to False to enable downloading (default: True)
#   - MISSING_CHECK_START_DATE / MISSING_CHECK_END_DATE: Date range to check
#   - Other settings (region, credentials, paths)
#
# Workflow:
#   1. First run with DRY_RUN=True: Lists missing files without downloading
#   2. Review output in logs
#   3. Set DRY_RUN=False in config.py
#   4. Re-submit to download missing files
#
# Output:
#   - Job logs: <PROJECT_ROOT>/logs/download_logs/imerg_missing_<jobid>.out
#   - Downloaded files: <PROJECT_ROOT>/data/imerg_data/
#
# =============================================================================

# -----------------------------------------------------------------------------
# SLURM Resource Configuration
# -----------------------------------------------------------------------------

#SBATCH --job-name=imerg_missing
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# Log paths (absolute paths required for SLURM directives)
#SBATCH --output=/home1/ppatel2025/nowcasting2/logs/download_logs/imerg_missing_%j.out
#SBATCH --error=/home1/ppatel2025/nowcasting2/logs/download_logs/imerg_missing_%j.err

# =============================================================================
# EXECUTION
# =============================================================================

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Project root is two levels up from script directory
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Create log directory if it doesn't exist
LOG_DIR="${PROJECT_ROOT}/logs/download_logs"
mkdir -p "$LOG_DIR"

# Initialize conda environment
source /home1/ppatel2025/miniconda3/etc/profile.d/conda.sh
conda activate tito_env

# Print configuration summary
echo "============================================================"
echo "IMERG Missing File Checker/Downloader"
echo "============================================================"
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Log directory: $LOG_DIR"
echo ""
echo "NOTE: Check DRY_RUN setting in config.py"
echo "      DRY_RUN=True  -> Only lists missing files"
echo "      DRY_RUN=False -> Downloads missing files"
echo "============================================================"

# Run the missing file checker/downloader from the script directory
cd "$SCRIPT_DIR"
python download_missing_imerg.py

echo "============================================================"
echo "Job completed."
echo "============================================================"
