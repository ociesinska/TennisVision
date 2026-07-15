import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mlflow

from tennisvision.core.mlflow_utils import setup_mlflow
from tennisvision.tasks.video_detection.postprocessing import TrackPostProcessingConfig, postprocess_tracking_result
from tennisvision.tasks.video_detection.tracking import compute_tracking_summary
from tennisvision.tasks.video_detection.types import VideoTrackDetection, VideoTrackingResult
from tennisvision.tasks.video_detection.visualization import render_tracking_video


def main():

    parser = argparse.ArgumentParser(description="Apply postprocessing to video tracking results.")
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tracking-uri", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--mlflow-experiment-name", type=str, default="Player Video Tracking")
    parser.add_argument("--max-tracks", type=int, default=TrackPostProcessingConfig.max_tracks)
    parser.add_argument("--min-count", type=int, default=TrackPostProcessingConfig.min_count)
    parser.add_argument("--min-presence-ratio", type=float, default=TrackPostProcessingConfig.min_presence_ratio)
    parser.add_argument("--max-stitch-frame-gap", type=int, default=TrackPostProcessingConfig.max_stitch_frame_gap)
    parser.add_argument("--max-stitch-gap-ratio", type=float, default=TrackPostProcessingConfig.max_stitch_gap_ratio)
    parser.add_argument("--max-stitch-overlap", type=int, default=TrackPostProcessingConfig.max_stitch_overlap)
    parser.add_argument("--max-stitch-overlap-ratio", type=float, default=TrackPostProcessingConfig.max_stitch_overlap_ratio)
    parser.add_argument("--max-stitch-center-distance", type=float, default=TrackPostProcessingConfig.max_stitch_center_distance)
    parser.add_argument(
        "--max-stitch-center-distance-ratio",
        type=float,
        default=TrackPostProcessingConfig.max_stitch_center_distance_ratio,
    )
    parser.add_argument("--max-stitch-area-ratio", type=float, default=TrackPostProcessingConfig.max_stitch_area_ratio)
    parser.add_argument("--edge-margin-ratio", type=float, default=TrackPostProcessingConfig.edge_margin_ratio)
    parser.add_argument("--max-reentry-frame-gap", type=int, default=TrackPostProcessingConfig.max_reentry_frame_gap)
    parser.add_argument("--max-reentry-gap-ratio", type=float, default=TrackPostProcessingConfig.max_reentry_gap_ratio)
    parser.add_argument("--max-reentry-center-distance", type=float, default=TrackPostProcessingConfig.max_reentry_center_distance)
    parser.add_argument(
        "--max-reentry-center-distance-ratio",
        type=float,
        default=TrackPostProcessingConfig.max_reentry_center_distance_ratio,
    )
    parser.add_argument("--max-reentry-area-ratio", type=float, default=TrackPostProcessingConfig.max_reentry_area_ratio)
    parser.add_argument("--reentry-side-ratio", type=float, default=TrackPostProcessingConfig.reentry_side_ratio)
    parser.add_argument("--min-duplicate-overlap-frames", type=int, default=TrackPostProcessingConfig.min_duplicate_overlap_frames)
    parser.add_argument("--min-duplicate-iou", type=float, default=TrackPostProcessingConfig.min_duplicate_iou)
    args = parser.parse_args()

    cfg = TrackPostProcessingConfig(
        min_count=args.min_count,
        min_presence_ratio=args.min_presence_ratio,
        max_stitch_frame_gap=args.max_stitch_frame_gap,
        max_stitch_gap_ratio=args.max_stitch_gap_ratio,
        max_stitch_overlap=args.max_stitch_overlap,
        max_stitch_overlap_ratio=args.max_stitch_overlap_ratio,
        max_stitch_center_distance=args.max_stitch_center_distance,
        max_stitch_center_distance_ratio=args.max_stitch_center_distance_ratio,
        max_stitch_area_ratio=args.max_stitch_area_ratio,
        edge_margin_ratio=args.edge_margin_ratio,
        max_reentry_frame_gap=args.max_reentry_frame_gap,
        max_reentry_gap_ratio=args.max_reentry_gap_ratio,
        max_reentry_center_distance=args.max_reentry_center_distance,
        max_reentry_center_distance_ratio=args.max_reentry_center_distance_ratio,
        max_reentry_area_ratio=args.max_reentry_area_ratio,
        reentry_side_ratio=args.reentry_side_ratio,
        min_duplicate_overlap_frames=args.min_duplicate_overlap_frames,
        min_duplicate_iou=args.min_duplicate_iou,
        max_tracks=args.max_tracks,
    )

    with args.tracks.open() as f:
        raw_result = json.load(f)

    result = VideoTrackingResult(
        video_path=raw_result["video_path"],
        width=raw_result["width"],
        height=raw_result["height"],
        fps=raw_result["fps"],
        detections=[VideoTrackDetection(**detection) for detection in raw_result["detections"]],
    )

    video_path = args.video or Path(result.video_path)
    output_dir = args.output_dir or args.tracks.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_mlflow(
        experiment_name=args.mlflow_experiment_name,
        tracking_uri=args.tracking_uri,
        set_experiment=True,
    )

    run_name = f"{video_path.stem}_postprocessed_{datetime.now():%Y%m%d_%H%M%S}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("run_type", "postprocess_tracks")
        mlflow.set_tag("task", "video_detection")

        processed_result, postprocessing_info = postprocess_tracking_result(result, cfg)
        tracking_summary = compute_tracking_summary(processed_result)

        tracks_output_path = output_dir / "tracks_postprocessed.json"
        info_output_path = output_dir / "postprocessing_info.json"
        summary_output_path = output_dir / "summary_postprocessed.json"

        with tracks_output_path.open("w") as f:
            json.dump(asdict(processed_result), f, indent=2)

        with info_output_path.open("w") as f:
            json.dump(postprocessing_info, f, indent=2)

        with summary_output_path.open("w") as f:
            json.dump(tracking_summary, f, indent=2)

        rendered_video_path = render_tracking_video(
            video_path=video_path, tracking_result=processed_result, output_path=output_dir / "video_postprocessed.mp4"
        )

        mlflow.log_artifact(str(rendered_video_path), artifact_path="postprocessing")
        mlflow.log_artifact(str(tracks_output_path), artifact_path="tracking")
        mlflow.log_artifact(str(info_output_path), artifact_path="tracking")
        mlflow.log_artifact(str(summary_output_path), artifact_path="tracking")
        mlflow.log_metrics(tracking_summary)


if __name__ == "__main__":
    main()
