# Project architecture

[Documentation index](README.md)

TennisVision is a Python package with task-specific pipelines and shared utilities.
It currently provides image classification, image detection, and video tracking;
court calibration, ball tracking, and match analytics are future components.

## Repository layout

```text
src/tennisvision/
├── core/                       # MLflow, device selection, visualization, explainability
├── scripts/                    # Shared model-registry utilities
└── tasks/
    ├── shot_classification/    # Image datasets, training, evaluation, and API
    ├── detection/              # Detector backends, evaluation, and image API
    └── video_detection/        # Video adapter, tracking, postprocessing, and API
tests/
├── unit/video_detection/
└── integration/video_detection/
docs/                           # Guides and experiment narratives
└── assets/video-tracking/       # Small visual examples referenced by docs
```

Local data, checkpoints, and MLflow artifacts live outside the documentation tree.
The `src` package layout and console commands are configured in
[`pyproject.toml`](../pyproject.toml).

## Pipeline responsibilities

| Task | Input and flow | Output |
| --- | --- | --- |
| Shot classification | Class-folder images → split/transforms → pretrained classifier → training or evaluation | Image labels/probabilities, checkpoints, metrics, Grad-CAM images |
| Detection | Image → detector loader/backend → prediction/evaluation | Bounding boxes, class labels, confidence, overlays, detection metrics |
| Video tracking | Video → detector + tracker → frame detections → postprocessing | Track IDs and boxes, tracking JSON, summaries, rendered video |

The video task reuses the detection model loader. Its
[`types.py`](../src/tennisvision/tasks/video_detection/types.py) defines the result
dataclasses; [`postprocessing.py`](../src/tennisvision/tasks/video_detection/postprocessing.py)
contains deterministic cleanup operations. Model/video adapters remain separate from
that logic. Frame metadata travels with results through processing and serialization.

Shot classification currently classifies individual images. It does not yet detect
shot events in a video or consume player tracks as a temporal classifier.

## Entry points and configuration

Each task has CLI scripts under `scripts/` and a separate FastAPI application under
`api/`. Training and evaluation use configuration dataclasses and CLI arguments;
the APIs read environment variables during startup. See task guides for the variables
actually consumed by each entry point. Settings modules exist, but not all entry
points use them yet, so names are not fully uniform across tasks.

[`core/mlflow_utils.py`](../src/tennisvision/core/mlflow_utils.py) handles shared
experiment setup and classification model loading. Detection backends handle their
own model artifacts. MLflow stores experiment metrics and artifacts; the Markdown
experiment records explain the motivation, observations, and decisions behind runs.

## Development boundaries

Pure calculations use synthetic tests. Integration tests exercise selected file and
pipeline boundaries while replacing external services. Model evaluation needs
datasets and checkpoints and is kept separate from ordinary CI.

The next planned stages are quantitative tracking evaluation, then court localization
and homography-based player projection. Pixel coordinates and processed-frame indices
are the current tracking representation; world-coordinate metrics are not implemented.
