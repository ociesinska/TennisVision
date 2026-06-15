# TennisVision

TennisVision is a modular computer vision project for tennis analytics. It currently contains two task areas:

- shot classification
- player/object detection

The project uses a `src/` layout, MLflow for experiment tracking, and FastAPI for serving task-specific inference APIs.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project in editable mode:

```bash
uv pip install -e .
```

If you do not install the project, prefix commands with `PYTHONPATH=src`.

## MLflow

Start the MLflow server:

```bash
make mlflow
```

The default tracking URI used by the scripts and APIs is:

```text
http://127.0.0.1:8080
```

## FastAPI Apps

The APIs are task-specific. Run them separately while the project is still evolving.

### Shot Classification API

```bash
uv run uvicorn tennisvision.tasks.shot_classification.api.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
```

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"
```

### Detection API

```bash
uv run uvicorn tennisvision.tasks.detection.api.app:app \
  --host 0.0.0.0 \
  --port 8001 \
  --reload
```

Detection models can be loaded from MLflow or from a local checkpoint during development. MLflow has priority when `DETECTION_MODEL_URI` or `DETECTION_RUN_ID` is set.

Configure a source MLflow run:

```bash
export DETECTION_RUN_ID="<run_id>"
```

or a full MLflow artifact URI:

```bash
export DETECTION_MODEL_URI="runs:/<run_id>/models/best.pt"
```

The default artifact path for `DETECTION_RUN_ID` is:

```bash
export DETECTION_MODEL_ARTIFACT_PATH="models/best.pt"
```

For local development, leave `DETECTION_MODEL_URI` and `DETECTION_RUN_ID` empty and set:

```bash
export DETECTION_MODEL_PATH="data/artifacts/detection/yolo11n_baseline/weights/best.pt"
```

Example JSON prediction request:

```bash
curl -X POST "http://localhost:8001/predict" \
  -F "file=@image.jpg"
```

Example visualization request:

```bash
curl -X POST "http://localhost:8001/predict/image" \
  -F "file=@image.jpg" \
  --output detections.png
```

## Detection CLI

Run detection inference with a model stored in MLflow:

```bash
uv run python -m tennisvision.tasks.detection.scripts.infer \
  --image data/detection_test/images/test1.png \
  --run-id <run_id> \
  --tracking-uri http://127.0.0.1:8080 \
  --visualize true
```

You can also pass a full MLflow model artifact URI:

```bash
uv run python -m tennisvision.tasks.detection.scripts.infer \
  --image data/detection_test/images/test1.png \
  --model-uri runs:/<run_id>/models/best.pt \
  --tracking-uri http://127.0.0.1:8080 \
  --visualize true
```

Run detection inference with a local checkpoint during development:

```bash
uv run python -m tennisvision.tasks.detection.scripts.infer \
  --image data/detection_test/images/test1.png \
  --model-path data/artifacts/detection/yolo11n_baseline/weights/best.pt \
  --visualize true
```

## Project Structure

```text
src/tennisvision/
├── core/
│   ├── mlflow_utils.py
│   ├── mlflow_cli.py
│   ├── utils.py
│   └── viz.py
├── scripts/
│   └── set_alias.py
└── tasks/
    ├── detection/
    │   ├── api/
    │   ├── backends/
    │   ├── scripts/
    │   ├── data.py
    │   ├── experiments.py
    │   ├── inference.py
    │   ├── types.py
    │   └── visualization.py
    └── shot_classification/
        ├── api/
        ├── scripts/
        ├── data.py
        ├── engine.py
        ├── experiments.py
        └── models.py
```

## Troubleshooting

If Python cannot import `tennisvision`, install the project:

```bash
uv pip install -e .
```

or run commands with:

```bash
PYTHONPATH=src
```

If a port is already in use:

```bash
lsof -i :8000
```

Then stop the relevant process or choose another port.
