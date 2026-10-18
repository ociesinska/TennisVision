import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeSettings:
    mlflow_tracking_uri: str
    device: str


def load_runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080"), device=os.getenv("DEVICE", "auto"))
