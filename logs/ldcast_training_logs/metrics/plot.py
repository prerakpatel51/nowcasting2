import pandas as pd
import matplotlib.pyplot as plt

base = "/home1/ppatel2025/nowcasting2/logs/ldcast_training_logs/metrics"

df1 = pd.read_csv(f"{base}/version_3_smallest/metrics.csv")
df1 = df1[df1["epoch"] > 0]

fig, ax = plt.subplots(figsize=(10, 6))

# Version 1
ax.plot(df1["epoch"], df1["train_loss"], label="V3 Train Loss", color="tab:blue")
ax.plot(df1["epoch"], df1["val_loss"], label="V3 Val Loss", color="tab:blue", linestyle="--")

ax.set_title("LDCast Training: Version 3", fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{base}/version_3.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to {base}/version.png")
