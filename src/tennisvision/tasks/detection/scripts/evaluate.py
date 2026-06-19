import argparse
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mlflow

from tennisvision.core.mlflow_utils import _jsonable, setup_mlflow
from tennisvision.tasks.detection.backends.ultralytics_yolo import (
    extract_ultralytics_metrics,
    log_ultralytics_eval_to_mlflow,
)
from tennisvision.tasks.detection.evaluation import DetectionEvaluationConfig, evaluate_detector
from tennisvision.tasks.detection.inference import get_model_source

logger = logging.getLogger(__name__)


def make_run_name(cfg: DetectionEvaluationConfig) -> str:
    if cfg.run_id:
        model_name = f"run_{cfg.run_id[:8]}"
    elif cfg.model_uri:
        model_name = cfg.model_uri.replace(":/", "_").replace("/", "_").replace(":", "_")
        model_name = model_name[-60:]
    else:
        model_name = cfg.model_path.stem

    return f"eval_{cfg.split}_{model_name}_{datetime.now():%Y%m%d_%H%M%S}"


def make_model_tag(cfg: DetectionEvaluationConfig, model_tag: str | None) -> str:
    if model_tag:
        return model_tag

    if cfg.run_id:
        return f"run_{cfg.run_id[:8]}"

    if cfg.model_uri:
        return cfg.model_uri

    return cfg.model_path.stem


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation.")
    parser.add_argument("--data-config", type=Path, default=Path("data/detection/data.yaml"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model-path", type=Path, default=DetectionEvaluationConfig.model_path)
    parser.add_argument("--backend", type=str, default="ultralytics", choices=["ultralytics", "torchvision"])
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=DetectionEvaluationConfig.batch)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-uri", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--model-artifact-path",
        type=str,
        default=DetectionEvaluationConfig.model_artifact_path,
    )
    parser.add_argument("--tracking-uri", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--mlflow-experiment-name", type=str, default="TennisVisionDetectionEvaluation")
    parser.add_argument("--dataset-tag", type=str, default=None)
    parser.add_argument("--model-tag", type=str, default=None)
    args = parser.parse_args()

    cfg = DetectionEvaluationConfig(
        backend=args.backend,
        data_config=args.data_config,
        split=args.split,
        model_path=args.model_path,
        model_uri=args.model_uri,
        run_id=args.run_id,
        model_artifact_path=args.model_artifact_path,
        tracking_uri=args.tracking_uri,
        imgsz=args.imgsz,
        batch=args.batch,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
    )

    setup_mlflow(
        experiment_name=args.mlflow_experiment_name,
        tracking_uri=args.tracking_uri,
        set_experiment=True,
    )

    run_name = make_run_name(cfg)
    with mlflow.start_run(run_name=run_name):
        model_tag = make_model_tag(cfg, args.model_tag)

        mlflow.set_tag("run_type", "evaluation")
        mlflow.set_tag("task", "detection")
        mlflow.set_tag("backend", cfg.backend)
        mlflow.set_tag("model", model_tag)
        mlflow.set_tag("split", cfg.split)

        if args.dataset_tag:
            mlflow.set_tag("dataset", args.dataset_tag)

        mlflow.log_dict(_jsonable(asdict(cfg)), "evaluation/config.json")

        params = {
            "data_config": cfg.data_config,
            "split": cfg.split,
            "model_source": get_model_source(cfg),
            "imgsz": cfg.imgsz,
            "batch": cfg.batch,
            "confidence": cfg.confidence,
            "iou": cfg.iou,
            "device": cfg.device,
        }

        if cfg.run_id is not None:
            params["source_run_id"] = cfg.run_id
            params["model_artifact_path"] = cfg.model_artifact_path
        elif cfg.model_uri is not None:
            params["model_uri"] = cfg.model_uri
        else:
            params["model_path"] = cfg.model_path

        mlflow.log_params(_jsonable(params))

        logger.info("Running model evaluation...")
        results = evaluate_detector(cfg)

        if cfg.backend == "ultralytics":
            metrics = extract_ultralytics_metrics(results)
            mlflow.log_metrics(metrics)
            log_ultralytics_eval_to_mlflow(results, cfg, metrics)
            logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
