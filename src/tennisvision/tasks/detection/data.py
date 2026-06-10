from tennisvision.tasks.detection.experiments import DetectionExperimentConfig


def validate_inputs(data_config: DetectionExperimentConfig) -> None:
    if not data_config.exists():
        raise FileNotFoundError(f"No config file for the dataset in {data_config.project_dir}.")
 