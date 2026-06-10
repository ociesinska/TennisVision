from dataclasses import dataclass
from pathlib import Path

from tennisvision.tasks.detection.backends.torchvision_detection import run_torchvision_experiment
from tennisvision.tasks.detection.backends.ultralytics_yolo import run_ultralytics_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[5]

DATA_CONFIG = PROJECT_ROOT / "data" / "detection" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "data" / "artifacts" / "detection"


@dataclass
class DetectionExperimentConfig:
    data_config: Path
    backend: str = "ultralytics"
    model: str = "yolo11n.pt"
    run_name: str = "yolo11n_baseline"
    epochs: int = 50
    batch: int = 4
    imgsz: int = 960
    workers: int = 0
    seed: int = 42
    deterministic: bool = True
    project_dir: Path = Path("data/artifacts/detection")
    device: str = "auto"


def run_experiment(cfg: DetectionExperimentConfig):

    if cfg.backend == "ultralytics":
        return run_ultralytics_experiment(cfg)

    if cfg.backend == "torchvision":
        return run_torchvision_experiment(cfg)

    raise ValueError(f"Unknown backend: {cfg.backend}")