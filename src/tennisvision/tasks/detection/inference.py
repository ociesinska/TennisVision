from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tennisvision.tasks.detection.backends.torchvision_detection import (
    load_torchvision_detector,
    predict_torchvision_image,
)
from tennisvision.tasks.detection.backends.ultralytics_yolo import (
    load_ultralytics_detector,
    load_ultralytics_detector_from_mlflow,
    predict_ultralytics_image,
    viz_detected_boxes,
)
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
    visualize_dir: Path = Path("data/artifacts/detection_test_results")
    model_uri: str | None = None
    run_id: str | None = None
    tracking_uri: str | None = None


def load_detector(cfg: DetectionInferenceConfig):
    if cfg.backend == "ultralytics":
        if cfg.model_uri is not None or cfg.run_id is not None:
            return load_ultralytics_detector_from_mlflow(
                run_id=cfg.run_id,
                model_uri=cfg.model_uri,
                tracking_uri=cfg.tracking_uri,
            )
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
        image_stem = Path(image_path).stem
        save_path = cfg.visualize_dir / f"{image_stem}_detections.png"
        viz_detected_boxes(detection_results, save_path=save_path)
    return detection_results
