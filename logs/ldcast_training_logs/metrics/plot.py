import pandas as pd
import matplotlib.pyplot as plt

base = "/home1/ppatel2025/nowcasting2/logs/ldcast_training_logs/metrics"

df1 = pd.read_csv(f"{base}/version_0/metrics.csv")
df1 = df1[df1["epoch"] > 0]
df2 = pd.read_csv(f"{base}/version_2/metrics.csv")
df2 = df2[df2["epoch"] > 0]

fig, ax = plt.subplots(figsize=(10, 6))

# Version 1
ax.plot(df1["epoch"], df1["train_loss"], label="V1 Train Loss", color="tab:blue")
ax.plot(df1["epoch"], df1["val_loss"], label="V1 Val Loss", color="tab:blue", linestyle="--")

# Version 2
ax.plot(df2["epoch"], df2["train_loss"], label="V2 Train Loss", color="tab:orange")
ax.plot(df2["epoch"], df2["val_loss"], label="V2 Val Loss", color="tab:orange", linestyle="--")

ax.set_title("LDCast Training: Version 1 vs Version 2", fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{base}/comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to {base}/comparison.png")
