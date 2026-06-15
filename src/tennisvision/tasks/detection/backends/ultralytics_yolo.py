from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from tennisvision.core.utils import get_device
from tennisvision.tasks.detection.data import validate_inputs
from tennisvision.tasks.detection.types import Detection, DetectionResult

if TYPE_CHECKING:
    from tennisvision.tasks.detection.experiments import DetectionExperimentConfig
    from tennisvision.tasks.detection.inference import DetectionInferenceConfig


logger = logging.getLogger(__name__)


def run_ultralytics_experiment(cfg: DetectionExperimentConfig):
    from ultralytics import YOLO

    validate_inputs(cfg.data_config)

    device = get_device(cfg.device)

    model = YOLO(cfg.model)

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

    logger.info(f"Results saved in: {results.save_dir}.")

    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    logger.info(f"Best model: {best_model_path}")

    return {
        "save_dir": str(results.save_dir),
        "best_model_path": str(best_model_path),
        "results_csv": str(Path(results.save_dir) / "results.csv"),
    }


def load_ultralytics_detector(model_path: str | Path) -> Any:
    from ultralytics import YOLO

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

def viz_detected_boxes(
    detection_result: DetectionResult,
    save_path: str | Path | None = None,
):

    if not detection_result.image_path:
        raise ValueError("DetectionResult.image_path is required to visualize detection result.")

    image = Image.open(detection_result.image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for detection in detection_result.detections:
        x1, y1, x2, y2 = detection.xyxy
        label = f"{detection.label} {detection.confidence:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
        text_x = x1
        text_y = max(0, y1 - 12)
        draw.text((text_x, text_y), label, fill=(255, 0, 0), font=font)

    if save_path is not None:
        output_path = Path(save_path)
        if not output_path.suffix:
            image_name = Path(detection_result.image_path).stem
            output_path = output_path / f"{image_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    return image

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
