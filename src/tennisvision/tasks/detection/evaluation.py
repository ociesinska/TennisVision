from dataclasses import dataclass
from pathlib import Path

from tennisvision.tasks.detection.backends.ultralytics_yolo import evaluate_ultralytics_detector
from tennisvision.tasks.detection.inference import load_detector


@dataclass
class DetectionEvaluationConfig:
    backend: str = "ultralytics"
    data_config: Path = Path("data/detection/data.yaml")
    split: str = "test"

    model_path: Path = Path("data/artifacts/detection/yolo11n_baseline/weights/best.pt")
    model_uri: str | None = None
    run_id: str | None = None
    model_artifact_path: str = "models/best.pt"
    tracking_uri: str | None = None

    imgsz: int = 960
    batch: int = 8
    confidence: float = 0.25
    iou: float = 0.7
    device: str = "auto"



def evaluate_detector(cfg: DetectionEvaluationConfig):
    model = load_detector(cfg)
    if cfg.backend == "ultralytics":
        eval_results = evaluate_ultralytics_detector(model, cfg)

    elif cfg.backend == "torchvision":
        raise NotImplementedError(f"Evaluation not yet implemented for backend {cfg.backend}")

    return eval_results
