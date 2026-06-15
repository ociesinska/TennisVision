from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import mlflow

from tennisvision.core.mlflow_utils import _jsonable, setup_mlflow
from tennisvision.core.utils import seed_everything
from tennisvision.tasks.detection.backends.torchvision_detection import run_torchvision_experiment
from tennisvision.tasks.detection.backends.ultralytics_yolo import run_ultralytics_experiment, save_yolo_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[4]

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
    mlflow_experiment_name: str = "Player Detection"
    mlflow_tracking_uri: str = "http://127.0.0.1:8080"

    artifacts_dir: Path = Path("artifacts")
    mlflow_dir: Path = Path("artifacts/detection/mlflow")


def run_experiment(cfg: DetectionExperimentConfig):

    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    has_active_run = mlflow.active_run() is not None

    setup_mlflow(experiment_name=cfg.mlflow_experiment_name, tracking_uri=cfg.mlflow_tracking_uri, set_experiment=not has_active_run)

    run_name = f"{cfg.run_name}_{datetime.now():%Y%m%d_%H%M%S}"

    with mlflow.start_run(run_name=run_name, nested=has_active_run):
        mlflow.log_dict(_jsonable(asdict(cfg)), "config.json")

        if cfg.backend == "ultralytics":
            
            result =  run_ultralytics_experiment(cfg)

            artifacts = save_yolo_artifacts(result['save_dir'])
            for artifact_group, paths in  artifacts.items():
                for path in paths:
                    mlflow.log_artifact(str(path), artifact_path=artifact_group)
            
            return result
                

        if cfg.backend == "torchvision":
            result = run_torchvision_experiment(cfg)
            return result

        raise ValueError(f"Unknown backend: {cfg.backend}")
