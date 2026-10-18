# Player detection

[Documentation index](../README.md) · [Experiment history](../experiments/detection.md)

The detection task produces image-space boxes, labels, and confidence scores. The practical baseline uses YOLO; a torchvision backend also exists. See the experiment history for dataset iterations, measured results, and the selected model.

## Image inference

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

## API

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

For the recorded YOLO runs, explicitly set the artifact path to `weights/best.pt` (the API code currently defaults to `models/best.pt`):

```bash
export DETECTION_MODEL_ARTIFACT_PATH="weights/best.pt"
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

## Training and evaluation

Use the [experiment record](../experiments/detection.md) for the dataset history and
reproduction commands. Training uses a YOLO data YAML supplied through `--data-config`.
The scripts live under [detection/scripts](../../src/tennisvision/tasks/detection/scripts).
Use `--help` on `run_experiment`, `evaluate`, or `infer` to inspect their accepted flags.

A detection benchmark measures box/class prediction quality. Video identity quality
is assessed separately in the [tracking workflow](video-tracking.md).
