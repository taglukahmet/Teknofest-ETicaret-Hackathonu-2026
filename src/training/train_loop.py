from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.config import config
from src.mlops.wandb_tracking import (
    finish_wandb_run,
    init_wandb_run,
    log_wandb_metrics,
)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _labels_from_batch(batch: dict[str, Any]) -> torch.Tensor:
    if "labels" in batch:
        return batch["labels"].long()
    if "label" in batch:
        return batch["label"].long()
    raise KeyError("Batch must contain a 'labels' or 'label' tensor.")


def _forward_logits(model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    model_inputs = {"input_ids": batch["input_ids"]}
    if "attention_mask" in batch:
        model_inputs["attention_mask"] = batch["attention_mask"]
    if "token_type_ids" in batch:
        model_inputs["token_type_ids"] = batch["token_type_ids"]

    outputs = model(**model_inputs)
    return outputs.logits if hasattr(outputs, "logits") else outputs


def _evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in val_loader:
            batch = _move_batch_to_device(batch, device)
            labels = _labels_from_batch(batch)
            logits = _forward_logits(model, batch)

            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            losses.append(float(loss.detach().cpu()))
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_predictions.extend(predictions.detach().cpu().numpy().tolist())

    macro_f1 = (
        f1_score(
            np.asarray(all_labels),
            np.asarray(all_predictions),
            average="macro",
            zero_division=0,
        )
        if all_labels
        else 0.0
    )

    return {
        "val_loss": float(np.mean(losses)) if losses else 0.0,
        "val_macro_f1": float(macro_f1),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = config.EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    device: str | torch.device | None = None,
    use_wandb: bool = False,
    wandb_project: str = "teknofest-eticaret-2026",
    wandb_run_name: str | None = None,
    wandb_mode: str | None = None,
) -> dict[str, list[float]]:
    """
    (Tasks: Member 3 & 5)
    Executes the forward/backward passes. 
    Must contain W&B tracking hooks for logging loss and validation metrics.
    """
    target_device = _resolve_device(device)
    model.to(target_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    run = init_wandb_run(
        project=wandb_project,
        run_name=wandb_run_name,
        mode=wandb_mode,
        enabled=use_wandb,
        config_values={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "device": str(target_device),
            "train_batch_size": getattr(train_loader, "batch_size", None),
            "val_batch_size": getattr(val_loader, "batch_size", None),
        },
        tags=["training"],
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
    }

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []

            for batch in train_loader:
                batch = _move_batch_to_device(batch, target_device)
                labels = _labels_from_batch(batch)
                logits = _forward_logits(model, batch)

                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            validation_metrics = _evaluate_model(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=target_device,
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(validation_metrics["val_loss"])
            history["val_macro_f1"].append(validation_metrics["val_macro_f1"])

            log_wandb_metrics(
                run,
                {
                    "train_loss": train_loss,
                    **validation_metrics,
                },
                step=epoch,
            )

    finally:
        finish_wandb_run(run)

    return history
