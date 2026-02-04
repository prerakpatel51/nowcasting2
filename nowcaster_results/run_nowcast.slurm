#!/bin/bash
# =============================================================================
# SLURM Job Script: Nowcasting Model Inference
# =============================================================================
#
# Description:
#   Runs inference for precipitation nowcasting models (STEPS and Lagrangian
#   Persistence) on specified dataset splits and saves predictions to HDF5.
#
# Usage:
#   # Run all models on test set (default)
#   sbatch run_nowcast.slurm
#
#   # Run specific model on specific split (edit variables below)
#   sbatch run_nowcast.slurm
#
# Output:
#   - Results: nowcaster_results/forecast_results/{model}/{split}/results.h5
#   - Job logs: logs/nowcast_logs/run_nowcast_<jobid>.out
#
# =============================================================================

# -----------------------------------------------------------------------------
# SLURM Resource Configuration
# -----------------------------------------------------------------------------

#SBATCH --job-name=run_nowcast
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/home1/ppatel2025/nowcasting2/logs/nowcast_logs/run_nowcast_%j.out
#SBATCH --error=/home1/ppatel2025/nowcasting2/logs/nowcast_logs/run_nowcast_%j.err

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

SCRIPT_DIR="/home1/ppatel2025/nowcasting2/nowcaster_results"
PROJECT_ROOT="/home1/ppatel2025/nowcasting2"

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
# MODEL: 'steps', 'lp', or 'all' (default: all)
# SPLIT: 'train', 'val', or 'test' (default: test)

MODEL="all"
SPLIT="test"

# =============================================================================
# EXECUTION
# =============================================================================

source /home1/ppatel2025/miniconda3/etc/profile.d/conda.sh
conda activate tito_env

# Print configuration summary
echo "============================================================"
echo "Nowcasting Model Inference"
echo "============================================================"
echo "Job ID:           $SLURM_JOB_ID"
echo "Node:             $SLURMD_NODENAME"
echo "Script directory: $SCRIPT_DIR"
echo "Project root:     $PROJECT_ROOT"
echo "Model:            $MODEL"
echo "Split:            $SPLIT"
echo "Start time:       $(date)"
echo "============================================================"

# Run the nowcasting script
cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/run_nowcast.py" --model "$MODEL" --split "$SPLIT"

echo "============================================================"
echo "Job completed at: $(date)"
echo "============================================================"
