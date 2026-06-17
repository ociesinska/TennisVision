from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tennisvision.tasks.detection.backends.torchvision_detection import (
    load_torchvision_detector,
    predict_torchvision_image,
)
from tennisvision.tasks.detection.backends.ultralytics_yolo import (
    load_ultralytics_detector,
    load_ultralytics_detector_from_mlflow,
    predict_ultralytics_image,
)
from tennisvision.tasks.detection.types import DetectionResult


@dataclass
class DetectorLoadConfig(Protocol):
    backend: str
    model_path: Path
    model_uri: str | None
    run_id: str | None
    model_artifact_path: str
    tracking_uri: str | None
    device: str


@dataclass
class DetectionInferenceConfig:
    backend: str = "ultralytics"
    model_path: Path = Path("data/artifacts/detection/yolo11n_baseline/weights/best.pt")
    model_artifact_path: str = "models/best.pt"
    imgsz: int = 960
    confidence: float = 0.25
    iou: float = 0.7
    device: str = "auto"
    visualize: bool = False
    model_uri: str | None = None
    run_id: str | None = None
    tracking_uri: str | None = None


def uses_mlflow_model(cfg: DetectionInferenceConfig) -> bool:
    return cfg.model_uri is not None or cfg.run_id is not None


def get_model_source(cfg: DetectionInferenceConfig) -> str:
    if cfg.model_uri is not None and cfg.run_id is not None:
        raise ValueError("Provide either model_uri or run_id, not both.")

    if cfg.model_uri is not None:
        return cfg.model_uri

    if cfg.run_id is not None:
        return f"runs:/{cfg.run_id}/{cfg.model_artifact_path}"

    return str(cfg.model_path)


def load_detector(cfg: DetectionInferenceConfig):
    if uses_mlflow_model(cfg):
        if cfg.backend != "ultralytics":
            raise NotImplementedError("MLflow model loading is currently implemented only for ultralytics backend.")

        return load_ultralytics_detector_from_mlflow(
            run_id=cfg.run_id,
            model_uri=cfg.model_uri,
            tracking_uri=cfg.tracking_uri,
            artifact_path=cfg.model_artifact_path,
        )

    if cfg.backend == "ultralytics":
        return load_ultralytics_detector(cfg.model_path)

    if cfg.backend == "torchvision":
        return load_torchvision_detector(cfg.model_path, device=cfg.device)

    raise ValueError(f"Unknown backend: {cfg.backend}")


def predict_image(model, image_path: str | Path, cfg: DetectionInferenceConfig) -> DetectionResult:
    if cfg.backend == "ultralytics":
        detection_results = predict_ultralytics_image(model, image_path, cfg)
    elif cfg.backend == "torchvision":
        detection_results = predict_torchvision_image(model, image_path, cfg)
    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")

    return detection_results
