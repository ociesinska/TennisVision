from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import torch

from tennisvision.core.engine import History


def setup_mlflow(
    experiment_name: str,
    tracking_uri: str | None = None,
    set_experiment: bool = True,
    artifact_root: str | None = None,
) -> None:
    """
    Mlflow configuration:
    - tracking_uri: sqlite:///artifacts/mlflow/mlruns.db
    - artifact_root: file:./artifacts/mlflow/artifacts
    """
    # Defaults
    experiment_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "TennisVision")
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///artifacts/mlflow/mlruns.db")

    mlflow.set_tracking_uri(tracking_uri)
    if set_experiment:
        mlflow.set_experiment(experiment_name)

def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    """Changes Path/torch.device etc. to serialized values to JSON"""
    out: dict[str, Any] = {}

    for k, v in d.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, torch.device):
            out[k] = str(v)
        else:
            out[k] = v
    return out

def log_config(cfg: Any) -> None:
    mlflow.log_dict(_jsonable(asdict(cfg)), "config.json")


def mlflow_log_history(hist: History, prefix: str = "") -> None:
    for i, v in enumerate(hist.train_loss, start=1):
        mlflow.log_metric(f"{prefix}train/loss", float(v), step=i)
    for i, v in enumerate(hist.val_loss, start=1):
        mlflow.log_metric(f"{prefix}val/loss", float(v), step=i)
    for i, v in enumerate(hist.train_metric, start=1):
        mlflow.log_metric(f"{prefix}train/metric", float(v), step=i)
    for i, v in enumerate(hist.val_metric, start=1):
        mlflow.log_metric(f"{prefix}val/metric", float(v), step=i)

    if "lr" in hist:
        for epoch, lr in enumerate(hist["lr"], start=1):
            mlflow.log_metric(f"{prefix}lr", float(lr), step=epoch)

    mlflow.log_metric(f"{prefix}best/val_metric", float(hist.best_val_metric))
    mlflow.log_metric(f"{prefix}best/epoch", float(hist.best_epoch))