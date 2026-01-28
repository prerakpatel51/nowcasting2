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
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# -----------------------------------------------------------------------------
# Determine paths relative to this script's location
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="${PROJECT_ROOT}/logs/download_logs"
mkdir -p "$LOG_DIR"

# Redirect output to log directory
exec > "${LOG_DIR}/hdf5_convert_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/hdf5_convert_${SLURM_JOB_ID}.err"

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

# Read conda config from Python config file
CONDA_PATH=$(python3 -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from config import CONDA_PATH; print(CONDA_PATH)")
CONDA_ENV=$(python3 -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from config import CONDA_ENV; print(CONDA_ENV)")

# Initialize conda environment
source "$CONDA_PATH"
conda activate "$CONDA_ENV"

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
