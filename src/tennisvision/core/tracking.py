from __future__ import annotations

import os

import mlflow


def setup_mlflow(
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
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
    mlflow.set_experiment(experiment_name)
