#!/bin/bash
#SBATCH --job-name=grand_denoising
#SBATCH --output=grand_denoising_%j.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --licenses=sps
#SBATCH --dependency=singleton

# Load modules and environment
module purge
source /pbs/throng/grand/soft/miniconda3/etc/profile.d/conda.sh
conda activate grandio
cd /pbs/home/s/selbouch/grand
source env/setup.sh

# Memory configuration
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Checkpoint management
CHECKPOINT_DIR="./output/checkpoints"
mkdir -p $CHECKPOINT_DIR

# Find latest checkpoint
LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/checkpoint_epoch_*.pt 2>/dev/null | head -n 1)

# Training command with checkpoint resumption
TRAIN_CMD="python /pbs/home/s/selbouch/grand_project/train_grand.py"
if [ -n "$LATEST_CHECKPOINT" ]; then
    TRAIN_CMD+=" --resume $LATEST_CHECKPOINT"
fi


MAX_TIME_SEC=$((24*3600 - 300))

timeout $MAX_TIME_SEC $TRAIN_CMD


if [ $? -eq 124 ]; then
    echo "Time limit approaching - resubmitting job..."
    LATEST_EPOCH=$(ls $CHECKPOINT_DIR/checkpoint_epoch_*.pt | grep -oE '[0-9]+' | sort -nr | head -n1)
    
    if [ -n "$LATEST_EPOCH" ]; then
        echo "Resuming from epoch $LATEST_EPOCH"
        sbatch --dependency=singleton $0
    else
        echo "No checkpoint found - restarting from scratch"
        sbatch $0
    fi
else
    echo "Training completed successfully"
fi
