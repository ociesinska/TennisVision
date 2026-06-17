from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mlflow

from tennisvision.core.mlflow_utils import _jsonable, setup_mlflow
from tennisvision.tasks.detection.inference import (
    DetectionInferenceConfig,
    get_model_source,
    load_detector,
    predict_image,
)
from tennisvision.tasks.detection.visualization import viz_detected_boxes

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detection inference on one image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DetectionInferenceConfig.model_path)
    parser.add_argument(
        "--backend",
        type=str,
        default="ultralytics",
        choices=["ultralytics", "torchvision"],
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-uri", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--model-artifact-path",
        type=str,
        default=DetectionInferenceConfig.model_artifact_path,
    )
    parser.add_argument("--tracking-uri", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--mlflow-experiment-name", type=str, default="TennisVisionDetectionInference")
    parser.add_argument(
        "--visualize",
        type=lambda value: str(value).lower() in {"1", "true", "yes", "y"},
        default=False,
    )
    parser.add_argument("--visualize-dir", type=Path, default=Path("data/artifacts/detection_test_results"))
    args = parser.parse_args()

    cfg = DetectionInferenceConfig(
        backend=args.backend,
        model_path=args.model_path,
        model_artifact_path=args.model_artifact_path,
        imgsz=args.imgsz,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
        visualize=args.visualize,
        model_uri=args.model_uri,
        run_id=args.run_id,
        tracking_uri=args.tracking_uri,
    )

    setup_mlflow(
        experiment_name=args.mlflow_experiment_name,
        tracking_uri=args.tracking_uri,
        set_experiment=True,
    )

    model = load_detector(cfg)

    image_stem = args.image.stem
    run_name = f"infer_{image_stem}_{datetime.now():%Y%m%d_%H%M%S}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("run_type", "inference")
        mlflow.set_tag("task", "detection")
        mlflow.set_tag("backend", cfg.backend)

        logger.info("Running inference...")

        result = predict_image(model, args.image, cfg)
        result_dict = asdict(result)

        mlflow.log_params(
            _jsonable(
                {
                    "image_path": args.image,
                    "model_source": get_model_source(cfg),
                    "model_path": cfg.model_path,
                    "model_uri": cfg.model_uri,
                    "source_run_id": cfg.run_id,
                    "model_artifact_path": cfg.model_artifact_path,
                    "imgsz": cfg.imgsz,
                    "confidence": cfg.confidence,
                    "iou": cfg.iou,
                    "device": cfg.device,
                    "visualize": cfg.visualize,
                }
            )
        )

        mlflow.log_metric("num_detections", len(result.detections))

        if result.detections:
            mean_confidence = sum(detection.confidence for detection in result.detections) / len(result.detections)
            mlflow.log_metric("mean_confidence", mean_confidence)

        mlflow.log_dict(result_dict, f"predictions/{image_stem}.json")

        if cfg.visualize:
            image_stem = args.image.stem
            visualization_path = args.visualize_dir / f"{image_stem}_detections.png"
            viz_detected_boxes(result, save_path=visualization_path)

            if visualization_path.exists():
                mlflow.log_artifact(str(visualization_path), artifact_path="visualizations")


if __name__ == "__main__":
    main()
