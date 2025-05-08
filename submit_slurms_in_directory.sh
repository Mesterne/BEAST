#!/bin/bash

DIR=${1:-.}

for file in "$DIR"/*.slurm; do
  if [[ -f "$file" ]]; then
    echo "Submitting $file..."
    sbatch "$file"
  else
    echo "No .slurm files found in $DIR."
    break
  fi
done
