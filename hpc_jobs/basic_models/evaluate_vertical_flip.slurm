#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=vertical_flip
#SBATCH --time=0-09:00:00

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --nodes=1

#SBATCH --output=idun_outputs/%x_%j/output.txt
#SBATCH --error=idun_outputs/%x_%j/output.err

WORKDIR=${SLURM_SUBMIT_DIR}
run_dir="${WORKDIR}/idun_outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${run_dir}"

echo "Running from directory: ${WORKDIR}"
echo "Name of job: ${SLURM_JOB_NAME}"
echo "ID of job: ${SLURM_JOB_ID}"
echo "Nodes used: ${SLURM_JOB_NODELIST}"
echo "Output directory: ${run_dir}"

# Activate environment
module load Anaconda3/2024.02-1
conda activate IBUN_BEAST

# Set output directory environment variable for the Python script
export OUTPUT_DIR="${run_dir}"
export JOB_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

# Run Python script from the parent directory
cd "${WORKDIR}"
python src/main.py 'experiments/gridloss/basic_models/config_vertical_flip.yml'
