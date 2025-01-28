# 🐉 BEAST: Multivariate Time Series What-If Scenario Generator

Welcome to **BEAST**! This powerful tool lets you generate insightful "what-if" scenarios for multivariate time series datasets. Whether you're analyzing climate patterns, financial trends, or sensor data, BEAST empowers your data explorations. 🚀

## 🌟 Features
- **Flexible Scenario Modeling:** Simulate hypothetical scenarios with ease.
- **Multivariate Support:** Handle complex time series data.
- **User-Friendly Interface:** Intuitive setup for data scientists.

---

## 📦 Installation

Clone the repository and create the Conda environment:

```bash
git clone https://github.com/Mesterne/BEAST.git
cd BEAST
conda env create -f BEAST_ENV.yml
conda activate my_environment
```

---

## 🚀 Usage

### 🔒 Setting Up Pre-Commit Hook
To automatically format and lint your code before committing:

1. Copy the pre-commit hook script to the Git hooks directory:

   ```bash
   cp precommit_hook.sh .git/hooks/pre-commit
   chmod +x .git/hooks/pre-commit
   ```

2. Now, every time you commit, Black and Flake8 checks will run automatically.

### Code Quality Checks
Ensure code quality by running:

```bash
flake8 .
black .
```


---

## 🛠️ Development

To contribute to BEAST, install development dependencies:

```bash
conda install black flake8
```

Ensure code quality by running:

```bash
flake8 .
black .
```

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🤝 Contributions

We welcome contributions from the community! Feel free to open issues and submit pull requests.

---

## 📧 Contact

For questions or support, please contact us at [your-email@example.com](mailto:your-email@example.com).

Happy data exploring! 🌍📊

