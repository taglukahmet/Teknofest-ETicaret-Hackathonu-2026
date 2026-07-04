from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from src.config import config


def build_default_run_config(extra_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """
    Builds the config payload that W&B stores with each experiment run.
    """
    run_config = asdict(config)
    if extra_config:
        run_config.update(dict(extra_config))
    return run_config


def init_wandb_run(
    project: str = "teknofest-eticaret-2026",
    run_name: str | None = None,
    config_values: Mapping[str, Any] | None = None,
    tags: list[str] | None = None,
    mode: str | None = None,
    enabled: bool = True,
) -> Any | None:
    """
    Starts a W&B run.

    Set enabled=False for local smoke tests. Set mode="offline" or mode="disabled"
    when you want to test logging without sending anything to the cloud.
    """
    if not enabled:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Install project requirements or run "
            "`pip install wandb==0.17.0`."
        ) from exc

    return wandb.init(
        project=project,
        name=run_name,
        config=build_default_run_config(config_values),
        tags=tags,
        mode=mode,
    )


def log_wandb_metrics(
    run: Any | None,
    metrics: Mapping[str, Any],
    step: int | None = None,
    prefix: str | None = None,
) -> None:
    """
    Logs a metric dictionary to W&B if a run is active.
    """
    if run is None:
        return

    payload = {}
    for key, value in metrics.items():
        if value is None:
            continue
        metric_name = f"{prefix}/{key}" if prefix else key
        payload[metric_name] = value

    if payload:
        run.log(payload, step=step)


def finish_wandb_run(run: Any | None) -> None:
    """
    Finishes a W&B run if one is active.
    """
    if run is not None:
        run.finish()
