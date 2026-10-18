# Video tracking

[Documentation index](../README.md) · [Experiment history](../experiments/video-tracking.md)

The pipeline runs player detection with ByteTrack, then cleans track identities and optionally selects active match players. The experiment record contains the tracker comparison, four postprocessing iterations, and qualitative clip reviews.

## Run tracking

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

Tracking statistics use the number of processed frames, including frames without
detections. Frame counts are preserved in `tracks.json` and through postprocessing;
the source video's reported count is stored separately for reference. Files without
a processed count use an explicitly labelled detection-span estimate.

The same outputs are logged to the `Player Video Tracking` MLflow experiment. Large videos should usually stay as local/MLflow artifacts rather than committed to git; use small screenshots or short GIFs in documentation when a visual preview is needed.

## Video Tracking Postprocessing

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

## Video API

Set a model source before starting the service. For a local checkpoint:

```bash
export VIDEO_DETECTION_MODEL_PATH="data/artifacts/detection/yolo11s_baseline/weights/best.pt"
uv run uvicorn tennisvision.tasks.video_detection.api.app:app --port 8002 --reload
```

For MLflow, use `VIDEO_DETECTION_RUN_ID` or `VIDEO_DETECTION_MODEL_URI` instead;
`VIDEO_DETECTION_MODEL_ARTIFACT_PATH` defaults to `weights/best.pt`. Avoid setting
both MLflow source options at once. Clear those variables when switching to a local
checkpoint. The service also reads `MLFLOW_TRACKING_URI` and `DEVICE`.

Use `/docs` for the `/track_postprocess` upload form and its postprocessing options. The response
includes detections and a URL for the rendered video, served from
`/results/{result_id}/video.mp4`. Check `/health` for model readiness.

## Pipeline and limits

Postprocessing filters short tracks, merges overlapping duplicates, stitches
fragments, deduplicates frame/ID pairs, then selects active tracks. Its deterministic
logic is covered by [unit and integration tests](../development.md).

The current clip reviews are qualitative; quantitative tracking evaluation on
annotated clips is a next step. Court calibration and world-coordinate movement
statistics are not part of the implemented tracking pipeline yet.
