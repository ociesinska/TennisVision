from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tennisvision.tasks.detection.backends.torchvision_detection import load_torchvision_detector, predict_torchvision_image
from tennisvision.tasks.detection.backends.ultralytics_yolo import load_ultralytics_detector, predict_ultralytics_image, viz_detected_boxes
from tennisvision.tasks.detection.types import DetectionResult


@dataclass
class DetectionInferenceConfig:
    backend: str = "ultralytics"
    model_path: Path = Path("data/artifacts/detection/yolo11n_baseline/weights/best.pt")
    imgsz: int = 960
    confidence: float = 0.25
    iou: float = 0.7
    device: str = "auto"
    visualize: bool = False


def load_detector(cfg: DetectionInferenceConfig):
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

    if cfg.visualize:
        viz_detected_boxes(
            detection_results,
            Path("data/artifacts/detection_test_results"),
        )

    return detection_results
