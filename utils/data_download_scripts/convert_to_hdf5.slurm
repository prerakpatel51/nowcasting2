#!/bin/bash
# =============================================================================
# SLURM Job Script: IMERG GeoTIFF to HDF5 Converter
# =============================================================================
#
# Description:
#   Converts IMERG precipitation GeoTIFF files to HDF5 format optimized for
#   deep learning dataloaders. Processes files in batches to avoid memory issues.
#
# Usage:
#   sbatch convert_to_hdf5.slurm
#
# Configuration:
#   All parameters are set in config.py. Environment variables below can
#   override defaults if needed.
#
# Output:
#   - HDF5 file: <PROJECT_ROOT>/data/imerg_data.h5
#   - Job logs: <PROJECT_ROOT>/logs/download_logs/hdf5_convert_<jobid>.out
#
# =============================================================================

# -----------------------------------------------------------------------------
# SLURM Resource Configuration
# -----------------------------------------------------------------------------

#SBATCH --job-name=imerg_hdf5
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/home1/ppatel2025/nowcasting2/logs/download_logs/hdf5_convert_%j.out
#SBATCH --error=/home1/ppatel2025/nowcasting2/logs/download_logs/hdf5_convert_%j.err

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

# Output directory for HDF5 file
# export HDF5_OUTPUT_DIR="${PROJECT_ROOT}/data/"

# HDF5 output filename
# export HDF5_FILENAME="imerg_data.h5"

# Batch size for processing (number of files per batch)
# export HDF5_BATCH_SIZE="1000"

# =============================================================================
# EXECUTION
# =============================================================================


source /home1/ppatel2025/miniconda3/etc/profile.d/conda.sh
conda activate tito_env

# Print configuration summary
echo "============================================================"
echo "IMERG GeoTIFF to HDF5 Converter"
echo "============================================================"
echo "Job ID:           $SLURM_JOB_ID"
echo "Node:             $SLURMD_NODENAME"
echo "Script directory: $SCRIPT_DIR"
echo "Project root:     $PROJECT_ROOT"
echo "Log directory:    $LOG_DIR"
echo "Start time:       $(date)"
echo "============================================================"

# Run the conversion script
cd "$SCRIPT_DIR"
python convert_to_hdf5.py

echo "============================================================"
echo "Job completed at: $(date)"
echo "============================================================"
