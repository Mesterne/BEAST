## 📊 BEAST Data Structure Guide

This document outlines the structure of data flowing within BEAST, ensuring consistency and clarity in processing.

---

## 🔹 Features
Features should be stored as a **NumPy array (`np.ndarray`)** with the following shape:

**`(Number of timeseries, Number of features)`**

### 🏷️ Feature Order
The array must follow this specific order:
```python
[
    "grid1-load_trend-strength",
    "grid1-load_trend-slope",
    "grid1-load_trend-linearity",
    "grid1-load_seasonal-strength",
    "grid1-loss_trend-strength",
    "grid1-loss_trend-slope",
    "grid1-loss_trend-linearity",
    "grid1-loss_seasonal-strength",
    "grid1-temp_trend-strength",
    "grid1-temp_trend-slope",
    "grid1-temp_trend-linearity",
    "grid1-temp_seasonal-strength",
]
```
📌 **Note:** These constants are defined in `src.data.constants.py`.

---

## 📈 Full Multivariate Time Series (MTS)
The full MTS should be a **NumPy array (`np.ndarray`)** with the following shape:

**`(Number of timeseries, Number of time steps * Number of UTS in an MTS)`**

### 📌 Time Series Order
Ensure the following order is maintained:
```python
[
    "grid1-load_t1",
    "grid1-load_t2",
    "grid1-load_t3",
    "grid1-load_t4",
    ...
    "grid1-loss_t1",
    "grid1-loss_t2",
    "grid1-loss_t3",
    "grid1-loss_t4",
    ...
    "grid1-temp_t1",
    "grid1-temp_t2",
    "grid1-temp_t3",
    "grid1-temp_t4",
]
```

## Model inputs (X, y) structures
All X arrays follows the following shape `(Number of samples, Number of features * Number of uts in a mts + Delta values of features (one for each feature) + One hot encoding (One for each uts in MTS))`

The result is the following structure:
```python
[
    "grid1-load_trend-strength",
    "grid1-load_trend-slope",
    "grid1-load_trend-linearity",
    "grid1-load_seasonal-strength",
    "grid1-loss_trend-strength",
    "grid1-loss_trend-slope",
    "grid1-loss_trend-linearity",
    "grid1-loss_seasonal-strength",
    "grid1-temp_trend-strength",
    "grid1-temp_trend-slope",
    "grid1-temp_trend-linearity",
    "grid1-temp_seasonal-strength",
    "delta_trend-strength",
    "delta_trend-slope",
    "delta_trend-linearity",
    "delta_seasonal-strength",
    "is_grid1-load_delta",
    "is_grid1-loss_delta",
    "is_grid1-temp_delta",
]
```

