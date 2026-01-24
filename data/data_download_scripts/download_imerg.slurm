#!/bin/bash
#SBATCH --job-name=imerg_download
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home1/ppatel2025/ldcast/data/logs/imerg_download_%j.out
#SBATCH --error=/home1/ppatel2025/ldcast/data/logs/imerg_download_%j.err

# =============================================================================
# CONFIGURATION - Edit these variables as needed
# =============================================================================

# Region bounds (longitude/latitude) - Burkina Faso
export XMIN="-5.5"      # Western longitude
export XMAX="2.5"       # Eastern longitude
export YMIN="9.0"       # Southern latitude
export YMAX="17.0"      # Northern latitude

# Output grid size
export OUTPUT_HEIGHT="64"
export OUTPUT_WIDTH="64"

# Time period (format: YYYY-MM-DD-HH-MM)
export START_DATE="2014-06-01-00-00"
export END_DATE="2022-12-31-00-00"

# Output folder
export OUTPUT_FOLDER="/home1/ppatel2025/ldcast/data/imerg_data/"

# NASA PPS credentials
export EMAIL="aaravamudan2014@my.fit.edu"

# Server path
export SERVER_PATH="https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/"

# =============================================================================
# RUN SCRIPT
# =============================================================================

# Load any required modules 

source /home1/ppatel2025/miniconda3/etc/profile.d/conda.sh
conda activate tito_env

SCRIPT_ROOT="/home1/ppatel2025/ldcast/data/data_download_scripts"
cd $SCRIPT_ROOT



echo "Starting IMERG download..."
echo "Region: XMIN=$XMIN, XMAX=$XMAX, YMIN=$YMIN, YMAX=$YMAX"
echo "Grid size: ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT}"
echo "Period: $START_DATE to $END_DATE"
echo "Output: $OUTPUT_FOLDER"

python download_imerg_data.py

echo "Job completed."
