import os
import time

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
import wandb

from src.config import config


@torch.no_grad()
def _evaluate(model: nn.Module, val_loader: DataLoader, loss_fn: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    for batch in val_loader:
        labels = batch.pop("labels").to(device, non_blocking=True)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        logits = model(**batch)
        total_loss += loss_fn(logits, labels).item()

        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(val_loader)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, f1, np.array(all_probs), np.array(all_labels)


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
            labels = batch.pop("labels").to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad()
            with autocast(device.type, enabled=(device.type == "cuda")):
                logits = model(**batch)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            global_step += 1
            if global_step % 50 == 0:
                wandb.log({"train/loss": loss.item(), "train/step": global_step, "epoch": epoch})

        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_f1, val_probs, val_labels = _evaluate(model, val_loader, loss_fn, device)

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
            # Out-of-fold probabilities, for Member 5's threshold optimization / ensemble step.
            np.save(os.path.join(config.OUTPUT_DIR, f"oof_probs{suffix}.npy"), val_probs)
            np.save(os.path.join(config.OUTPUT_DIR, f"oof_labels{suffix}.npy"), val_labels)

    wandb.finish()
