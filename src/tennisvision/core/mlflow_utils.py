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
    - tracking_uri: "http://127.0.0.1:8080"
    - artifact_root: file:./artifacts/mlflow/artifacts
    """
    # Defaults
    experiment_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "TennisVision")
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")

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


def load_model_from_mlflow(
    *,
    run_id: str | None = None,
    model_uri: str | None = None,
    model_artifact_path: str | None = "model",
    tracking_uri: str | None = None,
    eval_mode: bool = True,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load a PyTorch model logged in MLflow.

    Exactly one of:
    - run_id  -> loads from runs:/<run_id>/<model_artifact_path>
    - model_uri -> loads from any MLflow URI, e.g. models:/name/1

    """

    if (run_id is None) == (model_uri is None):
        raise ValueError("Provide exactly one of: run_id or model_uri.")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if model_uri is None:
        model_uri = f"runs:/{run_id}/{model_artifact_path}"

    model = mlflow.pytorch.load_model(model_uri, map_location=torch.device("cpu"))
    model = model.to(device)

    if eval_mode:
        model.eval()

    return model, model_uri
