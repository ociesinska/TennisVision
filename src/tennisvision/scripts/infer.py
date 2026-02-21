# TODO: load model form MLFlow and run inference
import json
import logging
from datetime import datetime

import mlflow
import torch
from mlflow.tracking import MlflowClient

from tennisvision.core.data import Split, build_loaders, build_transforms
from tennisvision.core.engine import evaluate_split, log_eval_to_mlflow, plot_confusion_matrix, predict_loader
from tennisvision.core.mlflow_utils import load_model_from_mlflow, setup_mlflow
from tennisvision.core.utils import setup_logging

logger = logging.getLogger()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def main() -> None:

    setup_logging(logging.INFO)
    logger.info("Inference started.")

    tracking_uri = "http://127.0.0.1:8080"
    experiment_name = "TennisVisionInference"
    setup_mlflow(experiment_name=experiment_name, tracking_uri=tracking_uri, set_experiment=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    source_run_id = "454d5b0cb9d0429a8231ff4a522db6e4"
    client = MlflowClient(tracking_uri=tracking_uri)
    image_root = "data/Tennis positions/images"

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

    model, model_uri = load_model_from_mlflow(run_id=source_run_id, device=device)
    model.eval()

    image_root = "data/Tennis positions/images"

    # Create split and transforms for inference
    _, val_tfms = build_transforms(weights=None)  # Use default transforms

    # Build loaders (we only need test_loader for inference)
    (_, val_loader, test_loader), (_, _, _), classes = build_loaders(
        image_root=image_root,
        split=split,
        train_tfms=val_tfms,  # Use val transforms for all
        val_tmfs=val_tfms,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
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
        fig = plot_confusion_matrix(metrics["cm"], classes)
        logger.info(f"Metrics: {metrics}")

        log_eval_to_mlflow(metrics, fig, split_name="test")
        logger.info(f"Evaluation completed. test/acc={metrics['acc']:.4f}")


if __name__ == "__main__":
    main()
