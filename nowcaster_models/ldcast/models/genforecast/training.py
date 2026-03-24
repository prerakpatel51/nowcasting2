# Copied from LDCast paper: ldcast/models/genforecast/training.py. No modifications.

import os

import pytorch_lightning as pl
import torch

from ..diffusion import diffusion


def setup_genforecast_training(
    model,
    autoencoder,
    context_encoder,
    model_dir,
    lr=1e-4,
    lr_warmup=0,
    beta_schedule="linear",
):
    ldm = diffusion.LatentDiffusion(model, autoencoder,
        context_encoder=context_encoder, lr=lr, lr_warmup=lr_warmup,
        beta_schedule=beta_schedule)

    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if (num_gpus > 0) else "cpu"
    devices = torch.cuda.device_count() if (accelerator == "gpu") else 1
    num_nodes = int(os.environ.get("SLURM_NNODES", 1))

    early_stopping = pl.callbacks.EarlyStopping(
        "val_loss_ema", patience=7, verbose=True, check_finite=False
    )

    # Save best model by val_loss_ema + always keep last.ckpt for resuming
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="best-{epoch}-{val_loss_ema:.4f}",
        monitor="val_loss_ema",
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
        save_last=True,
    )

    # Save every epoch
    every_epoch_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="epoch-{epoch}-{val_loss_ema:.4f}",
        every_n_epochs=1,
        save_top_k=-1,  # keep all
    )

    # CSV logger for train/val loss per epoch
    csv_logger = pl.loggers.CSVLogger(
        save_dir="/home1/ppatel2025/nowcasting2/logs/ldcast_training_logs",
        name="metrics",
    )

    callbacks = [early_stopping, checkpoint, every_epoch_checkpoint]

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=1000,
        num_nodes=num_nodes,
        strategy='ddp' if (num_gpus > 1 or num_nodes > 1) else 'auto',
        callbacks=callbacks,
        logger=csv_logger,
        precision="32-true"
    )

    return (ldm, trainer)
