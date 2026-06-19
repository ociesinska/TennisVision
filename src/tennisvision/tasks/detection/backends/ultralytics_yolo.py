from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
from ultralytics import YOLO, settings

from tennisvision.core.utils import get_device
from tennisvision.tasks.detection.data import validate_inputs
from tennisvision.tasks.detection.types import Detection, DetectionResult

if TYPE_CHECKING:
    from tennisvision.tasks.detection.experiments import DetectionExperimentConfig
    from tennisvision.tasks.detection.inference import DetectionInferenceConfig


logger = logging.getLogger(__name__)


def run_ultralytics_experiment(cfg: DetectionExperimentConfig):

    validate_inputs(cfg.data_config)

    device = get_device(cfg.device)

    previous_mlflow_env = {
        "MLFLOW_EXPERIMENT_NAME": os.environ.get("MLFLOW_EXPERIMENT_NAME"),
        "MLFLOW_RUN": os.environ.get("MLFLOW_RUN"),
        "MLFLOW_KEEP_RUN_ACTIVE": os.environ.get("MLFLOW_KEEP_RUN_ACTIVE"),
    }

    settings.update({"mlflow": True})
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.mlflow_experiment_name
    os.environ["MLFLOW_RUN"] = cfg.run_name
    os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "True"

    model = YOLO(cfg.model)

    try:
        results = model.train(
            data=str(cfg.data_config),
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            device=device,
            workers=cfg.workers,
            project=str(cfg.project_dir),
            name=cfg.run_name,
            seed=cfg.seed,
            deterministic=cfg.deterministic,
        )
    finally:
        for key, value in previous_mlflow_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    logger.info(f"Results saved in: {results.save_dir}.")

    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    logger.info(f"Best model: {best_model_path}")

    return {
        "save_dir": str(results.save_dir),
        "best_model_path": str(best_model_path),
        "results_csv": str(Path(results.save_dir) / "results.csv"),
    }


def load_ultralytics_detector(model_path: str | Path) -> Any:

    return YOLO(str(model_path))


def predict_ultralytics_image(model: Any, image_path: str | Path, cfg: DetectionInferenceConfig):

    device = get_device(cfg.device)

    results = model.predict(
        source=str(image_path),
        conf=cfg.confidence,
        iou=cfg.iou,
        imgsz=cfg.imgsz,
        device=device,
        verbose=False,
    )

    result = results[0]
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                Detection(
                    class_id=class_id,
                    label=result.names[class_id],
                    confidence=confidence,
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                )
            )

    height, width = result.orig_shape

    return DetectionResult(
        image_path=str(image_path),
        width=int(width),
        height=int(height),
        detections=detections,
    )


def save_yolo_artifacts(save_dir: str | Path) -> dict[str, list[Path]]:
    save_dir = Path(save_dir)

    artifact_groups: dict[str, list[Path]] = {
        "models": [
            save_dir / "weights" / "best.pt",
            save_dir / "weights" / "last.pt",
        ],
        "training": [
            save_dir / "args.yaml",
            save_dir / "results.csv",
        ],
        "plots": [
            save_dir / "results.png",
            save_dir / "confusion_matrix.png",
            save_dir / "confusion_matrix_normalized.png",
            save_dir / "PR_curve.png",
            save_dir / "P_curve.png",
            save_dir / "R_curve.png",
            save_dir / "F1_curve.png",
        ],
    }

    existing_artifacts: dict[str, list[Path]] = {}
    for artifact_group, paths in artifact_groups.items():
        existing_paths = [path for path in paths if path.is_file()]

        if existing_paths:
            existing_artifacts[artifact_group] = existing_paths

    return existing_artifacts


def load_ultralytics_detector_from_mlflow(
    *,
    run_id: str | None = None,
    model_uri: str | None = None,
    tracking_uri: str | None = None,
    artifact_path: str = "models/best.pt",
) -> Any:

    if (run_id is None) == (model_uri is None):
        raise ValueError("Provide exactly one of: run_id or model_uri.")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if model_uri is None:
        model_uri = f"runs:/{run_id}/{artifact_path}"

    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    return YOLO(local_path)


def evaluate_ultralytics_detector(model, cfg):
    device = get_device(cfg.device)

    results = model.val(
        data=str(cfg.data_config),
        split=cfg.split,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        conf=cfg.confidence,
        iou=cfg.iou,
        device=device,
    )

    return results


def extract_ultralytics_metrics(results) -> dict[str, float]:
    return {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "map50": float(results.box.map50),
        "map75": float(results.box.map75),
        "map50_95": float(results.box.map),
        "fitness": float(results.fitness),
    }


def log_ultralytics_eval_artifacts(results):

    save_dir = Path(results.save_dir)
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".csv", ".txt", ".json"}

    for path in save_dir.iterdir():
        if path.is_file() and path.suffix.lower() in allowed_suffixes:
            mlflow.log_artifact(str(path), artifact_path="ultralytics_eval")


def log_ultralytics_eval_to_mlflow(results, cfg, metrics) -> None:
    mlflow.log_dict(results.results_dict, "evaluation/results_dict.json")

    summary = f"""# Detection Evaluation

        split: {cfg.split}
        mAP50: {metrics["map50"]:.4f}
        mAP50-95: {metrics["map50_95"]:.4f}
        precision: {metrics["precision"]:.4f}
        recall: {metrics["recall"]:.4f}
     """

    mlflow.log_text(summary, "evaluation/summary.md")
    log_ultralytics_eval_artifacts(results)
