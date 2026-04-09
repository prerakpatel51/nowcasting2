#!/bin/bash
#SBATCH --job-name=ldcast-train
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=11
#SBATCH --mem=500GB
#SBATCH --output=/home1/ppatel2025/nowcasting2/logs/ldcast_training_logs/overpred_ldcast_%j.log
#SBATCH --error=/home1/ppatel2025/nowcasting2/logs/ldcast_training_logs/overpred_ldcast_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tito_env

cd /home1/ppatel2025/nowcasting2

srun python -m nowcaster_models.ldcast.train --config /home1/ppatel2025/nowcasting2/nowcaster_models/ldcast/config_overpred.yaml  
