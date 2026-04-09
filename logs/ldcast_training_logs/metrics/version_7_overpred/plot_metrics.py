import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

csv_path = sys.argv[1] if len(sys.argv) > 1 else "metrics.csv"
save_dir = os.path.dirname(os.path.abspath(csv_path))

df = pd.read_csv(csv_path)

# Plot 1: Train loss and val loss vs epoch
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df["epoch"], df["train_loss"], label="Train Loss")
ax.plot(df["epoch"], df["val_loss"], label="Val Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Train Loss and Val Loss vs Epoch")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "loss_vs_epoch.png"), dpi=150)
plt.close(fig)

# Plot 2: Log(train_loss) and log(val_loss) vs epoch
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df["epoch"], np.log(df["train_loss"]), label="log(Train Loss)")
ax.plot(df["epoch"], np.log(df["val_loss"]), label="log(Val Loss)")
ax.set_xlabel("Epoch")
ax.set_ylabel("log(Loss)")
ax.set_title("Log(Train Loss) and Log(Val Loss) vs Epoch")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "log_loss_vs_epoch.png"), dpi=150)
plt.close(fig)

print(f"Saved plots to {save_dir}")
