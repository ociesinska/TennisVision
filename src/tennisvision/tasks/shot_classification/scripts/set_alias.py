from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:8080"
MODEL_NAME = "TennisVision"
VERSION = "1"
ALIAS = "champion"


def main():
    client = MlflowClient(tracking_uri=TRACKING_URI)
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=ALIAS,
        version=VERSION,
    )
    print(f"Set alias {MODEL_NAME}@{ALIAS} -> v{VERSION}")


if __name__ == "__main__":
    main()
