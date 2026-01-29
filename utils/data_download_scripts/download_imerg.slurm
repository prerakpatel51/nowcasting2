#!/bin/bash
# =============================================================================
# SLURM Job Script: IMERG Data Downloader
# =============================================================================
#
# Description:
#   Downloads NASA GPM IMERG precipitation data for a specified geographic
#   region and time period. Wraps download_imerg_data.py which reads all
#   configuration from config.py.
#
# Usage:
#   sbatch download_imerg.slurm
#
# Configuration:
#   All parameters (date range, region, credentials) are set in config.py.
#   Environment variables below can override config.py defaults if needed.
#
# Output:
#   - Processed GeoTIFF files: <PROJECT_ROOT>/data/imerg_data_tiff_clean/imerg.YYYYMMDDHHMM.tif
#   - Job logs: <PROJECT_ROOT>/logs/download_logs/imerg_download_<jobid>.out
#
# =============================================================================

# -----------------------------------------------------------------------------
# SLURM Resource Configuration
# -----------------------------------------------------------------------------

#SBATCH --job-name=imerg_download
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home1/ppatel2025/nowcasting2/logs/download_logs/imerg_download_%j.out
#SBATCH --error=/home1/ppatel2025/nowcasting2/logs/download_logs/imerg_download_%j.err

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

SCRIPT_DIR="/home1/ppatel2025/nowcasting2/utils/data_download_scripts"
PROJECT_ROOT="/home1/ppatel2025/nowcasting2"

# -----------------------------------------------------------------------------
# Optional Environment Variable Overrides
# -----------------------------------------------------------------------------
# Uncomment and modify these to override config.py defaults for this job.
# If not set, values from config.py will be used.

# Geographic bounds (decimal degrees)
# export XMIN="-5.5"
# export XMAX="2.5"
# export YMIN="9.0"
# export YMAX="17.0"

# Output grid dimensions (pixels)
# export OUTPUT_HEIGHT="64"
# export OUTPUT_WIDTH="64"

# Time period (format: YYYY-MM-DD-HH-MM)
# export START_DATE="2014-06-01-00-00"
# export END_DATE="2014-06-03-00-00"

# Output folder (defaults to PROJECT_ROOT/data/imerg_data_tiff_clean/)
# export OUTPUT_FOLDER="${PROJECT_ROOT}/data/imerg_data_tiff_clean/"

# NASA PPS credentials
# export EMAIL="your.email@example.com"

# =============================================================================
# EXECUTION
# =============================================================================

# Initialize conda environment
source /home1/ppatel2025/miniconda3/etc/profile.d/conda.sh
conda activate tito_env

# Print configuration summary
echo "============================================================"
echo "Starting IMERG download..."
echo "============================================================"
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "============================================================"

# Run the download script from the script directory
cd "$SCRIPT_DIR"
python download_imerg_data.py

echo "============================================================"
echo "Job completed."
echo "============================================================"
