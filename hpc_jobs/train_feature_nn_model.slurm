#!/bin/sh
#SBATCH --account=ie-idi       # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name=train_feed_forward_feature_model
#SBATCH --time=0-08:00:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1             # Setting the number of GPUs to 1
#SBATCH --mem=16G                 # Asking for 16GB RAM
#SBATCH --nodes=1
#SBATCH --output=run_logs/output.txt      # Specifying 'stdout'
#SBATCH --error=run_logs/output.err        # Specifying 'stderr'

#SBATCH --mail-user=vegasj@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
$current_run_id = $(date +%Y%m%d_%H%M%S)
mkdir current_run_id
cd current_run_id
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"


module purge

# Running your python file
module load Anaconda3/2024.02-1
conda activate BEAST_ENV
python ../../src/training/feedforward_features.py
