#!/bin/sh
#SBATCH --account=ie-idi
#SBATCH --job-name=train_feed_forward_feature_model
#SBATCH --time=0-23:00:00

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nodes=1

#SBATCH --output=idun_outputs/%x_%j/output.txt
#SBATCH --error=idun_outputs/%x_%j/output.err

#SBATCH --mail-user=vegasj@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
run_dir="${WORKDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "idun_outputs"
mkdir -p "idun_outputs/${run_dir}"

echo "Running from directory: ${WORKDIR}"
echo "Name of job: ${SLURM_JOB_NAME}"
echo "ID of job: ${SLURM_JOB_ID}"
echo "Nodes used: ${SLURM_JOB_NODELIST}"
echo "Output directory: ${run_dir}"

# Activate environment
module load Anaconda3/2024.02-1
conda activate IBUN_BEAST

# Set output directory environment variable for the Python script
export OUTPUT_DIR="idun_outputs/${run_dir}"

# Run Python script from the parent directory
cd "${WORKDIR}"
python src/training/feedforward_features.py
