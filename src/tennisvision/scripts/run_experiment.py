import logging

from tennisvision.core.experiments import ExperimentConfig, run_experiment
from tennisvision.core.utils import setup_logging

logger = logging.getLogger(__name__)


def main():

    setup_logging(logging.INFO)
    logger.info("Start training")

    cfg = ExperimentConfig(image_root="data/Tennis positions/images", model_name="resnet18", finetune=False)

    print(run_experiment(cfg))


if __name__ == "__main__":
    main()
