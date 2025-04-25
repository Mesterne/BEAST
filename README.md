# 🐉 BEASTS: Benchmarking Enhanced Analysis of Shifting Time Series

Welcome to **BEAST**! This powerful tool leverages a database of existing multivariate time series to generate new, realistic time series. By simulating distribution shifts, BEAST helps assess model robustness and ensures forecasting models improve after retraining.

---

## 🌟 Features

- **Realistic Distribution Shifts:** Generate time series data that mimic real-world changes.
- **Multivariate Support:** Handle complex datasets with multiple variables.
- **Scenario Simulation:** Test forecasting models against hypothetical shifts.
---

## 📦 Installation

Get started by cloning the repository and setting up the Conda environment:

```bash
git clone https://github.com/Mesterne/BEAST.git
cd BEAST
conda env create -f environment.yml
conda activate BEAST_ENV
```

---

## 🚀 Usage

### 🔧 Setting Up Pre-Commit Hook
To maintain code quality, set up a pre-commit hook that automatically formats and checks your code before committing:

1. Copy the pre-commit hook script:

   ```bash
   cp precommit_hook.sh .git/hooks/pre-commit
   chmod +x .git/hooks/pre-commit
   ```

2. Now, every time you commit, Black and Flake8 will run automatically to enforce coding standards.

### 🛠 Code Quality Checks
Manually check formatting and linting by running:

```bash
black .
```

---

## 🛠️ Development

To contribute to BEAST, install the necessary development dependencies:

```bash
conda install black
```

Ensure code consistency by running:

```bash
black .
```

---



