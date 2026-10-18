# Getting started

[Documentation index](README.md)

Run commands from the repository root. Use Python 3.11 or 3.12 (CI uses 3.12) and uv.

## Install

```bash
uv sync --locked --extra dev
source .venv/bin/activate
make check
```

This installs the project and its development tools using `uv.lock`. The current
installation includes the ML dependencies. Tests use synthetic inputs and do not
require model weights, datasets, a GPU, or an MLflow server. Inference and training
do require their task-specific data and model artifacts; pretrained weights may
need to be downloaded on first use.

If `UV_PROJECT_ENVIRONMENT` points to another directory, either unset it or use
`UV_PROJECT_ENVIRONMENT=.venv` consistently for installation and checks.

## Start MLflow

```bash
make mlflow
```

The local server is available at `http://127.0.0.1:8080`. The Makefile uses `./mlruns`
as its backend store and `./artifacts/mlflow/artifacts` for artifacts. Existing run IDs
in the experiment notes refer to the author's experiment store; a fresh checkout
does not include those models or datasets.

## Choose a task

- [Shot classification](tasks/shot-classification.md): image classes, training, evaluation, API, and Grad-CAM.
- [Player detection](tasks/detection.md): image inference, training, and evaluation.
- [Video tracking](tasks/video-tracking.md): tracking, postprocessing, and video API.

Each API is a separate process. The guides use ports 8000, 8001, and 8002 so they can
run alongside one another. Open `/docs` on the corresponding API for its request schema.
Set environment variables in the shell before starting the process; do not assume
that copying `.env.example` automatically loads configuration into every entry point.

## Troubleshooting

If imports fail, run `uv sync --locked --extra dev` from the repository root. For
source-only execution, `PYTHONPATH=src` makes the source package visible, but does
not install missing dependencies.

If pytest is missing despite an activated environment, check
`echo "$UV_PROJECT_ENVIRONMENT"`: uv may be targeting a different environment.

To inspect a port conflict, run `lsof -i :8000`; use another port or stop the process
that owns it. An unavailable model or MLflow run should be resolved using the model
source configuration in the relevant task guide.
