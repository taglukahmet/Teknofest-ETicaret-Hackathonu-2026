import os
import time

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import numpy as np
import polars as pl
import wandb

from src.config import config


def _logits_of(model_output):
    # architecture.py currently returns a raw tensor already, but this keeps the
    # loop working if a HF model output object (with a .logits attribute) is
    # passed in instead, without needing every caller to unwrap it themselves.
    return model_output.logits if hasattr(model_output, "logits") else model_output


@torch.no_grad()
def _evaluate(model: nn.Module, val_loader: DataLoader, loss_fn: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    all_ids, all_probs, all_preds, all_labels = [], [], [], []

    for batch in val_loader:
        ids = batch.pop("id")
        labels = batch.pop("labels").to(device, non_blocking=True)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        logits = _logits_of(model(**batch))
        total_loss += loss_fn(logits, labels).item()

        probs = torch.softmax(logits, dim=1)[:, 1]
        all_ids.extend(ids)
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(val_loader)
    # Macro-F1 to match the competition metric used in post_processing/metrics.py
    # (sklearn's default is binary F1, which would silently score something else).
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1, all_ids, np.array(all_probs), np.array(all_labels)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold_id: int | None = None,
) -> None:
    """
    (Tasks: Member 3 & 5)
    Executes the forward/backward passes.
    Must contain W&B tracking hooks for logging loss and validation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

    # Linear warmup (10% of steps) then linear decay: without this the first
    # optimizer steps at the full LR distort the pretrained weights before the
    # model has adapted, which looks like the model "forgetting" its pretraining.
    total_steps = max(1, config.EPOCHS * len(train_loader))
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    run_name = f"fold-{fold_id}" if fold_id is not None else None
    # WANDB_MODE=offline/disabled can be set in the shell if a teammate hasn't run `wandb login` yet.
    wandb.init(
        project="teknofest-eticaret-hackathonu",
        name=run_name,
        config={
            "model_name": config.MODEL_NAME,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "max_sequence_length": config.MAX_SEQUENCE_LENGTH,
            "fold_id": fold_id,
        },
    )
    wandb.watch(model, log="gradients", log_freq=500)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    suffix = f"_fold{fold_id}" if fold_id is not None else ""

    best_val_f1 = -1.0
    global_step = 0
    t0 = time.time()

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch.pop("id", None)
            labels = batch.pop("labels").to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad()
            with autocast(device.type, enabled=(device.type == "cuda")):
                logits = _logits_of(model(**batch))
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1
            if global_step % 50 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": global_step,
                    "epoch": epoch,
                })

        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_f1, val_ids, val_probs, val_labels = _evaluate(model, val_loader, loss_fn, device)

        wandb.log({
            "epoch": epoch,
            "train/epoch_loss": avg_train_loss,
            "val/loss": val_loss,
            "val/f1": val_f1,
            "elapsed_sec": time.time() - t0,
        })
        print(f"[epoch {epoch}] train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint_path = os.path.join(config.OUTPUT_DIR, f"best_model{suffix}.pt")
            torch.save(model.state_dict(), checkpoint_path)

            # OOF predictions for Member 5's ensemble (id, probability) and threshold
            # search (id, label) stages -- those read CSV/parquet via polars, not .npy.
            pl.DataFrame({"id": val_ids, "probability": val_probs}).write_csv(
                os.path.join(config.OUTPUT_DIR, f"oof_predictions{suffix}.csv")
            )
            pl.DataFrame({"id": val_ids, "label": val_labels}).write_csv(
                os.path.join(config.OUTPUT_DIR, f"oof_labels{suffix}.csv")
            )

    wandb.finish()
