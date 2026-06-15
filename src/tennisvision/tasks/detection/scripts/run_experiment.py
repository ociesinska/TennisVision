import argparse
import logging
from pathlib import Path

from tennisvision.core.utils import setup_logging
from tennisvision.tasks.detection.experiments import DATA_CONFIG, DetectionExperimentConfig, run_experiment

logger = logging.getLogger(__name__)


def main() -> None:

    parser = argparse.ArgumentParser(description="Run detection training.")
    parser.add_argument("--backend", type=str, default="ultralytics", choices=["ultralytics", "torchvision"])
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--data-config", type=Path, default=DATA_CONFIG)
    parser.add_argument("--run-name", type=str, default="yolo11n_baseline")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    cfg = DetectionExperimentConfig(
        backend=args.backend,
        model=args.model,
        data_config=args.data_config,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        run_name=args.run_name,
        workers=args.workers,
        seed=args.seed,
        deterministic=args.deterministic,
        device=args.device,
    )

    logger.info("Start training")

    run_experiment(cfg)


if __name__ == "__main__":
    main()
