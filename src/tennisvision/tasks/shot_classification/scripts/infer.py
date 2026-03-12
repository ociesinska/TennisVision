import argparse
import json
import logging
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

from tennisvision.core.data import Split, build_loaders, build_transforms
from tennisvision.core.engine import evaluate_split, log_eval_to_mlflow, plot_confusion_matrix, plot_random_misclassified_cases, predict_loader
from tennisvision.core.experiments import ExperimentConfig
from tennisvision.core.mlflow_utils import load_model_from_mlflow, setup_mlflow
from tennisvision.core.utils import get_device, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on test set")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run_id", type=str, help="MLflow source run ID")
    group.add_argument("--model_alias", type=str, help="Model alias (e.g., 'champion')")
    parser.add_argument("--model_name", type=str, default="TennisVision", help="Registered model name (used with --model_alias)")
    parser.add_argument("--image_root", type=str, default="data/Tennis positions/images", help="Path to image directory")
    args = parser.parse_args()

    cfg = ExperimentConfig(image_root=args.image_root)
    setup_logging(logging.INFO)
    logger.info("Inference started.")

    tracking_uri = cfg.mlflow_tracking_uri
    experiment_name = "TennisVisionInference"
    setup_mlflow(experiment_name=experiment_name, tracking_uri=tracking_uri, set_experiment=True)
    device = get_device()
    client = MlflowClient(tracking_uri=tracking_uri)

    # Resolve alias to run_id if using alias
    if args.model_alias:
        model_version = client.get_model_version_by_alias(args.model_name, args.model_alias)
        source_run_id = model_version.run_id
        model_uri = f"models:/{args.model_name}@{args.model_alias}"
        logger.info(f"Resolved {args.model_name}@{args.model_alias} -> version {model_version.version}, run_id={source_run_id}")
    else:
        source_run_id = args.run_id
        model_uri = None
    image_root = cfg.image_root
    split_path = client.download_artifacts(source_run_id, "split/indices.json")

    with open(split_path) as f:
        split_dict = json.load(f)

    idx_to_class_path = client.download_artifacts(source_run_id, "labels/idx_to_class.json")

    with open(idx_to_class_path) as x:
        idx_to_class = json.load(x)

    class_to_idx = {v: int(k) for k, v in idx_to_class.items()}

    split = Split(
        idx_train=split_dict["idx_train"],
        idx_val=split_dict["idx_val"],
        idx_test=split_dict["idx_test"],
        class_to_idx=class_to_idx,
        seed=split_dict["seed"],
    )

    model, model_uri = load_model_from_mlflow(
        run_id=source_run_id if not args.model_alias else None,
        model_uri=model_uri,
        device=device,
    )
    model.eval()

    # Create split and transforms for inference
    _, val_tfms = build_transforms(weights=None)  # Use default transforms

    # Build loaders (we only need test_loader for inference)
    (_, val_loader, test_loader), (_, _, _), classes = build_loaders(
        image_root=image_root,
        split=split,
        train_tfms=val_tfms,  # Use val transforms for all
        val_tfms=val_tfms,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    run_name = f"infer_{source_run_id[:8]}_{datetime.now():%Y%m%d_%H%M%S}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("run_type", "inference")
        mlflow.set_tag("source_model_run_id", source_run_id)
        mlflow.log_param("source_model_uri", model_uri)

        # Run inference
        logger.info("Running inference on test set...")
        predictions = predict_loader(model, test_loader, device)

        # Evaluate and log results
        metrics = evaluate_split(predictions, classes)
        cm = plot_confusion_matrix(metrics["cm"], classes)
        misclassified_plots = plot_random_misclassified_cases(predictions, test_loader, classes)
        logger.info(f"Metrics: {metrics}")

        log_eval_to_mlflow(metrics, cm, split_name="test")
        mlflow.log_figure(misclassified_plots, "misclassified samples/misclassified_samples.png")
        logger.info(f"Evaluation completed. test/acc={metrics['acc']:.4f}")


if __name__ == "__main__":
    main()
