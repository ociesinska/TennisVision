# TennisVision

TennisVision is an ongoing modular sports Computer Vision project for tennis broadcast analysis. It focuses on player detection, image and video inference, tracking, model evaluation, and experiment tracking.

The project currently includes:

- shot classification
- YOLO-based player/object detection
- image inference and visualization
- FastAPI inference APIs
- MLflow experiment tracking
- video player tracking with ByteTrack
- tracking postprocessing for ID cleanup, duplicate merging, re-entry stitching, and active-player selection

## Demo / Visual Preview

Example tracking output:

![Video tracking demo](experiments/video_tracking/assets/video1_det.gif)

Postprocessing example on a harder clip:

| Raw tracking | After postprocessing |
| --- | --- |
| ![Raw tracking before postprocessing](experiments/video_tracking/assets/video6_raw_4s_to_10s.gif) | ![Tracking after postprocessing](experiments/video_tracking/assets/video6_4s_to_10s_tracking_demo.gif) |

Postprocessing is a core part of the tracking methodology, not just a visualization cleanup step. Tennis videos require handling ID switches, short noisy tracks, duplicate identities, players leaving and re-entering the frame, neighboring-court players, staff/ball-kid false positives, and perspective-dependent active-player selection. The example above shows one representative case where postprocessing improves identity consistency after a difficult tracking event. Larger videos and local MLflow artifacts are intentionally not committed to the repository.

Demo GIFs are short processed clips with TennisVision overlays; source details are listed in [Third-Party Notices](THIRD_PARTY_NOTICES.md).

## Current Status

- YOLO-based player detection baseline trained and evaluated.
- Detection inference API implemented with FastAPI.
- Image inference and detection visualization implemented.
- Video tracking pipeline implemented with ByteTrack.
- Tennis-specific tracker configuration added.
- Tracking outputs are saved as annotated video, `tracks.json`, sample frames, and MLflow artifacts.
- Postprocessing implemented for short-track filtering, duplicate track merging, ID stitching, re-entry handling, and active-player selection.
- Project is ongoing; next steps focus on more robust main-court player selection and reducing identity issues across harder video perspectives.

## Key Results

Current practical detection baseline:

| Field | Value |
| --- | --- |
| Model | `yolo11s.pt` |
| Run | `yolo11s_img1280_ep50_playersv2` |
| mAP50 | `0.9427` |
| mAP50-95 | `0.7208` |
| Precision | `0.9421` |
| Recall | `0.8625` |

This is treated as a practical baseline, not a final detector. It was selected from the updated dataset iteration because it includes harder ball kid / staff cases observed during video testing.

## Why This Is Challenging

Tennis broadcast analysis is noisy in ways that are not visible from static benchmark metrics alone:

- players can be very small or distant;
- players are often partially occluded or clipped by the frame;
- overlays, scoreboards, logos, and broadcast artifacts change the image distribution;
- ball kids, referees, and staff can look similar to distant players;
- neighboring-court players may be valid tennis-player detections but not active match players;
- tracking can produce ID switches after occlusion, fast movement, or re-entry;
- a player may remain undetected for a few frames after re-entering the view;
- active-player selection depends on camera perspective.

## Detailed Experiment Notes

Detailed metrics, run IDs, configuration comparisons, qualitative observations, and tracking notes are kept in:

- [Detection experiments](experiments/detection/experiments.md)
- [Video tracking experiments](experiments/video_tracking/experiments.md)

## Running the Project

### Installation

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

### MLflow

Start the MLflow server:

```bash
make mlflow
```

The default tracking URI used by the scripts and APIs is:

```text
http://127.0.0.1:8080
```

### FastAPI Apps

The APIs are task-specific. Run them separately while the project is still evolving.

#### Shot Classification API

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

#### Detection API

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

### Detection CLI

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

### Video Tracking CLI

Track players in a video with a detection model stored in MLflow:

```bash
uv run python -m tennisvision.tasks.video_detection.scripts.track_video \
  --video data/videos/doubles1.mp4 \
  --run-id <run_id> \
  --model-artifact-path weights/best.pt \
  --imgsz 1280 \
  --confidence 0.5
```

Current recommended detector baseline:

```text
run_id: d21d335020d14f4498d5388aebebd03b
model_artifact_path: weights/best.pt
imgsz: 1280
confidence: 0.5
tracker: src/tennisvision/tasks/video_detection/configs/tracking/bytetrack_tennis.yaml
```

These values are intentionally explicit so tracking and postprocessing runs can be compared against a stable baseline.

The default tracker config is the project-specific ByteTrack config:

```text
src/tennisvision/tasks/video_detection/configs/tracking/bytetrack_tennis.yaml
```

The script writes local outputs to:

```text
data/artifacts/video_detection/<run_name>/
```

Typical outputs include:

- tracked video (`.mp4`)
- `tracks.json`
- sample frames for quick inspection

The same outputs are logged to the `Player Video Tracking` MLflow experiment. Large videos should usually stay as local/MLflow artifacts rather than committed to git; use small screenshots or short GIFs in documentation when a visual preview is needed.

#### Video Tracking Postprocessing

Postprocess existing `tracks.json` files to clean short tracks, merge duplicate IDs, stitch re-entry fragments, optionally select active players, and render a new annotated video.

Use the original raw video as `--video`. Passing an already annotated tracking output will draw labels on top of existing labels.

Recommended singles command:

```bash
uv run python -m tennisvision.tasks.video_detection.scripts.postprocess_tracks \
  --tracks data/artifacts/video_detection/<tracking_run>/tracks.json \
  --video data/public/videos/raw/<video_name>.mp4 \
  --min-duplicate-overlap-frames 5 \
  --min-duplicate-iou 0.5 \
  --max-stitch-frame-gap 120 \
  --max-stitch-gap-ratio 0.18 \
  --max-stitch-center-distance 250 \
  --max-stitch-center-distance-ratio 0.12 \
  --max-reentry-frame-gap 180 \
  --max-reentry-gap-ratio 0.25 \
  --max-reentry-center-distance 700 \
  --max-reentry-center-distance-ratio 0.35 \
  --reentry-side-ratio 0.4 \
  --max-tracks 2
```

For doubles, use:

```bash
--max-tracks 4
```

Postprocessing writes and logs:

- `tracks_postprocessed.json`
- `postprocessing_info.json`
- `summary_postprocessed.json`
- `video_postprocessed.mp4`

Known limitation: postprocessing can preserve identity after a player is detected again, but it does not create boxes for frames where the detector missed a heavily clipped or out-of-frame player.

### Troubleshooting

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
    ├── video_detection/
    │   ├── backends/
    │   ├── configs/
    │   ├── scripts/
    │   ├── postprocessing.py
    │   ├── tracking.py
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
