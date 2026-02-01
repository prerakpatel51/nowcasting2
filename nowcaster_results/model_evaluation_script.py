"""STEPS and Lagrangian Persistence model evaluation script."""

import numpy as np
from nowcaster_models.steps.config import DataConfig, STEPSConfig
from nowcaster_models.steps.dataloader import PrecipitationDataModule
from nowcaster_models.steps.steps import steps
from nowcaster_models.lagrangian_persistence.config import LPConfig
from nowcaster_models.lagrangian_persistence.lp import lagrangian_persistence


# Setup data
data_config = DataConfig(batch_size=1, num_workers=0)
dm = PrecipitationDataModule(config=data_config)
dm.setup(stage='test')

# Get a sample
test_loader = dm.test_dataloader()
inputs, targets = next(iter(test_loader))

# Extract input and target: (B, T, 1, H, W) -> (T, H, W)
input_seq = inputs[0].squeeze(1).numpy()
input_seq = np.nan_to_num(input_seq, nan=0.0)
target_seq = targets[0].squeeze(1).numpy()
target_seq = np.nan_to_num(target_seq, nan=0.0)

# Run STEPS forecast
steps_config = STEPSConfig()
forecast = steps(input_seq, timesteps=data_config.output_frames, config=steps_config)

print(f"Input shape: {input_seq.shape}")
print(f"Forecast shape: {forecast.shape}")
print(f"Target shape: {target_seq.shape}")

# Save images
import matplotlib.pyplot as plt

# Use consistent scale across all STEPS plots
vmin = 0
vmax = max(input_seq.max(), target_seq.max(), forecast.max())

# Save last input frame
plt.imshow(input_seq[-1], cmap='viridis', vmin=vmin, vmax=vmax)
plt.colorbar(label='mm/h')
plt.title('STEPS Input (last frame)')
plt.savefig('temp_input.png')
plt.close()

# Save first forecast frame
plt.imshow(forecast[0], cmap='viridis', vmin=vmin, vmax=vmax)
plt.colorbar(label='mm/h')
plt.title('STEPS Forecast (first frame)')
plt.savefig('temp_forecast.png')
plt.close()

# Save first target frame
plt.imshow(target_seq[0], cmap='viridis', vmin=vmin, vmax=vmax)
plt.colorbar(label='mm/h')
plt.title('STEPS Target (first frame)')
plt.savefig('temp_target.png')
plt.close()

print("Saved: temp_input.png, temp_forecast.png, temp_target.png")










# Lagrangian Persistence Evaluation
print("\n" + "="*50)
print("Lagrangian Persistence Evaluation")
print("="*50)

# Run Lagrangian Persistence forecast
lp_config = LPConfig()
lp_forecast = lagrangian_persistence(input_seq, timesteps=data_config.output_frames, config=lp_config)

print(f"Input shape: {input_seq.shape}")
print(f"LP Forecast shape: {lp_forecast.shape}")
print(f"Target shape: {targets[0].squeeze(1).shape}")

# Use consistent scale across all LP plots
vmin = 0
vmax = max(input_seq.max(), target_seq.max(), lp_forecast.max())

# Save last input frame
plt.imshow(input_seq[-1], cmap='viridis', vmin=vmin, vmax=vmax)
plt.colorbar(label='mm/h')
plt.title('LP Input (last frame)')
plt.savefig('temp_lp_input.png')
plt.close()

# Save first LP forecast frame
plt.imshow(lp_forecast[0], cmap='viridis', vmin=vmin, vmax=vmax)
plt.colorbar(label='mm/h')
plt.title('LP Forecast (first frame)')
plt.savefig('temp_lp_forecast.png')
plt.close()

# Save first target frame
plt.imshow(target_seq[0], cmap='viridis', vmin=vmin, vmax=vmax)
plt.colorbar(label='mm/h')
plt.title('LP Target (first frame)')
plt.savefig('temp_lp_target.png')
plt.close()

print("Saved: temp_lp_input.png, temp_lp_forecast.png, temp_lp_target.png")
