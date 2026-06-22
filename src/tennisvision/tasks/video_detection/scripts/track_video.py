import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mlflow

from tennisvision.core.mlflow_utils import setup_mlflow
from tennisvision.tasks.video_detection.tracking import VideoTrackingConfig, compute_tracking_summary, save_sample_frames, track_video


def main():
    parser = argparse.ArgumentParser(description="Track objects in a video.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default="Player Video Tracking",
    )
    parser.add_argument("--model-path", type=Path, default=VideoTrackingConfig.model_path)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--model-uri", type=str, default=None)
    parser.add_argument("--model-artifact-path", type=str, default=VideoTrackingConfig.model_artifact_path)
    parser.add_argument("--tracking-uri", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/artifacts/video_detection"))
    args = parser.parse_args()

    run_name = f"{args.video.stem}_{datetime.now():%Y%m%d_%H%M%S}"

    cfg = VideoTrackingConfig(
        model_path=args.model_path,
        model_uri=args.model_uri,
        run_id=args.run_id,
        model_artifact_path=args.model_artifact_path,
        tracking_uri=args.tracking_uri,
        imgsz=args.imgsz,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
        tracker=args.tracker,
        save_video=args.save_video,
        output_dir=args.output_dir.resolve(),
        run_name=run_name,
    )

    setup_mlflow(
        experiment_name=args.mlflow_experiment_name,
        tracking_uri=args.tracking_uri,
        set_experiment=True,
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("run_type", "video_tracking")
        mlflow.set_tag("task", "video_detection")
        mlflow.set_tag("tracker", cfg.tracker)

        result = track_video(cfg, args.video)

        output_dir = cfg.output_dir / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        tracks_path = output_dir / "tracks.json"

        with tracks_path.open("w") as f:
            json.dump(asdict(result), f, indent=2)

        mlflow.log_dict(asdict(cfg), "tracking/config.json")
        mlflow.log_artifact(str(output_dir), artifact_path="tracking")

        for video_path in output_dir.glob("*.mp4"):
            mlflow.log_artifact(str(video_path), artifact_path="video")

            for frame_path in save_sample_frames(video_path, output_dir):
                mlflow.log_artifact(str(frame_path), artifact_path="sample_frames")

        tracking_summary = compute_tracking_summary(result)
        mlflow.log_dict(tracking_summary, "tracking/summary.json")
        mlflow.log_metrics(tracking_summary)


if __name__ == "__main__":
    main()
