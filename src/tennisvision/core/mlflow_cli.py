import argparse
import os
from collections.abc import Sequence

from mlflow.tracking import MlflowClient


def set_registered_model_alias(*, tracking_uri: str, model_name: str, version: str, alias: str) -> None:
    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version,
    )
    print(f"Set alias {model_name}@{alias} -> v{version}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Set MLflow model alias")
    parser.add_argument(
        "--tracking_uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080"),
        help="MLflow tracking URI",
    )
    parser.add_argument("--model_name", type=str, default="TennisVision", help="Registered model name")
    parser.add_argument("--version", type=str, required=True, help="Model version to alias")
    parser.add_argument("--alias", type=str, default="champion", help="Alias to set")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    set_registered_model_alias(
        tracking_uri=args.tracking_uri,
        model_name=args.model_name,
        version=args.version,
        alias=args.alias,
    )


if __name__ == "__main__":
    main()
