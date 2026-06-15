from pathlib import Path


def validate_inputs(data_config: Path) -> None:
    if not data_config.exists():
        raise FileNotFoundError(f"No config file for the dataset in {data_config}.")
