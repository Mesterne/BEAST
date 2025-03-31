import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

# Set ggplot style
plt.style.use("ggplot")

# Load performance data from YAML file
with open("model_performances.yml", "r") as file:
    data = yaml.safe_load(file)

# Prepare DataFrame
df = pd.DataFrame(
    {
        "name": [v["name"] for v in data.values()],
        "performance": [v["performance"] for v in data.values()],
    }
)

# Determine colors
min_performance = df["performance"].min()
colors = [
    (
        "red"
        if p == min_performance
        else ("blue" if n == "Forecasting performance wo. retrain" else "gray")
    )
    for n, p in zip(df["name"], df["performance"])
]

# Plotting
plt.figure(figsize=(8, 5))
sns.barplot(y="performance", x="name", data=df, palette=colors)
plt.title("Model Performances", fontsize=16)
plt.xlabel("Model Name", fontsize=14)
plt.ylabel("Performance", fontsize=14)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
