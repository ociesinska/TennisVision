import argparse
import logging

from tennisvision.core.experiments import ExperimentConfig, run_experiment
from tennisvision.core.utils import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run training experiment")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model architecture")
    parser.add_argument("--image_root", type=str, default="data/Tennis positions/images", help="Path to image directory")
    parser.add_argument("--head_epochs", type=int, default=5, help="Head training epochs")
    parser.add_argument("--finetune_epochs", type=int, default=8, help="Finetune epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    logger.info("Start training")

    cfg = ExperimentConfig(
        image_root=args.image_root,
        model_name=args.model_name,
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        seed=args.seed,
    )

    run_experiment(cfg)


if __name__ == "__main__":
    main()
