import argparse
import os

from mlflow.tracking import MlflowClient


def main():
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
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=args.tracking_uri)
    client.set_registered_model_alias(
        name=args.model_name,
        alias=args.alias,
        version=args.version,
    )
    print(f"Set alias {args.model_name}@{args.alias} -> v{args.version}")


if __name__ == "__main__":
    main()
