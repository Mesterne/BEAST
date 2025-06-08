# Master Thesis: Hans Jakob Håland, Vegard Sjåvik

Welcome to the code used for the master thesis of Hans Jakob Håland and Vegard Sjåvik! This code leverages a database of existing multivariate time series to generate new, realistic time series.

---

## Installation

```bash
git clone https://github.com/Mesterne/BEAST.git
cd BEAST
conda env create -f environment.yml
conda activate BEAST_ENV
```

## Running experiments

### Locally

This is how to run the covariance model locally. To run another model simply change config_covariance_ga.yml with another .yml file in the experiments folder.

```bash
python src/main.py 'experiments/gridloss/two_stage/config_covariance_ga.yml'
```

### On IDUN

Most of the models from the thesis has been trained and evaluated using IDUN - A HPC cluster on NTNU. To run the an model here run:

```bash
sbatch hpc_jobs/two_stage/train_covariance_model_with_one_hot_encoding.slurm
```

To change which model to run. Just change the .slurm file with another one found in hpc_jobs.
