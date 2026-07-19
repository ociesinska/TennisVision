import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mlflow

from tennisvision.core.mlflow_utils import setup_mlflow
from tennisvision.tasks.video_detection.postprocessing import TrackPostProcessingConfig, compute_tracking_stats, postprocess_tracking_result
from tennisvision.tasks.video_detection.tracking import (
    VideoTrackingConfig,
    compute_tracking_summary,
    save_sample_frames,
    track_video,
)
from tennisvision.tasks.video_detection.visualization import render_tracking_video

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Track objects in a video and apply postprocessing.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mlflow-experiment-name", type=str, default="Player Video Tracking Pipeline")
    parser.add_argument("--model-path", type=Path, default=VideoTrackingConfig.model_path)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--model-uri", type=str, default=None)
    parser.add_argument("--model-artifact-path", type=str, default=VideoTrackingConfig.model_artifact_path)
    parser.add_argument("--tracking-uri", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tracker", type=str, default=VideoTrackingConfig.tracker)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/artifacts/video_tracking_pipeline"))
    parser.add_argument("--max-tracks", type=int, default=TrackPostProcessingConfig.max_tracks)
    parser.add_argument("--min-count", type=int, default=TrackPostProcessingConfig.min_count)
    parser.add_argument("--min-presence-ratio", type=float, default=TrackPostProcessingConfig.min_presence_ratio)
    parser.add_argument("--max-stitch-frame-gap", type=int, default=TrackPostProcessingConfig.max_stitch_frame_gap)
    parser.add_argument("--max-stitch-gap-ratio", type=float, default=TrackPostProcessingConfig.max_stitch_gap_ratio)
    parser.add_argument("--max-stitch-overlap", type=int, default=TrackPostProcessingConfig.max_stitch_overlap)
    parser.add_argument("--max-stitch-overlap-ratio", type=float, default=TrackPostProcessingConfig.max_stitch_overlap_ratio)
    parser.add_argument("--max-stitch-center-distance", type=float, default=TrackPostProcessingConfig.max_stitch_center_distance)
    parser.add_argument("--max-stitch-center-distance-ratio", type=float, default=TrackPostProcessingConfig.max_stitch_center_distance_ratio)
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

    run_name = f"{args.video.stem}_{datetime.now():%Y%m%d_%H%M%S}"

    tracking_cfg = VideoTrackingConfig(
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

    postprocessing_cfg = TrackPostProcessingConfig(
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

    setup_mlflow(
        experiment_name=args.mlflow_experiment_name,
        tracking_uri=args.tracking_uri,
        set_experiment=True,
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("run_type", "video_tracking_postprocessing")
        mlflow.set_tag("task", "video_detection_and_tracking")
        mlflow.set_tag("tracker", tracking_cfg.tracker)

        logger.info("Track objects on video.")
        result = track_video(tracking_cfg, args.video)

        output_dir = tracking_cfg.output_dir / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        tracks_path = output_dir / "tracks.json"

        with tracks_path.open("w") as f:
            json.dump(asdict(result), f, indent=2)

        mlflow.log_dict(asdict(tracking_cfg), "tracking/raw/config.json")
        mlflow.log_dict(asdict(postprocessing_cfg), "tracking/postprocessed/config.json")
        mlflow.log_artifact(str(tracks_path), artifact_path="tracking/raw")

        for raw_video_path in output_dir.glob("*.mp4"):
            mlflow.log_artifact(str(raw_video_path), artifact_path="video/raw")

            for frame_path in save_sample_frames(raw_video_path, output_dir):
                mlflow.log_artifact(str(frame_path), artifact_path="sample_frames/raw")

        tracking_summary = compute_tracking_summary(result)
        tracking_stats = compute_tracking_stats(result)
        mlflow.log_dict(tracking_summary, "tracking/raw/summary.json")
        mlflow.log_dict(tracking_stats, "tracking/raw/stats.json")
        mlflow.log_metrics({f"raw_{key}": value for key, value in tracking_summary.items()})

        logger.info("Apply video tracking postprocessing.")

        video_path = args.video or Path(result.video_path)

        processed_result, postprocessing_info = postprocess_tracking_result(result, postprocessing_cfg)
        postprocessed_tracking_summary = compute_tracking_summary(processed_result)

        tracks_output_path = output_dir / "tracks_postprocessed.json"
        info_output_path = output_dir / "postprocessing_info.json"
        summary_output_path = output_dir / "summary_postprocessed.json"

        with tracks_output_path.open("w") as f:
            json.dump(asdict(processed_result), f, indent=2)

        with info_output_path.open("w") as f:
            json.dump(postprocessing_info, f, indent=2)

        with summary_output_path.open("w") as f:
            json.dump(postprocessed_tracking_summary, f, indent=2)

        rendered_video_path = render_tracking_video(
            video_path=video_path, tracking_result=processed_result, output_path=output_dir / "video_postprocessed.mp4"
        )

        mlflow.log_artifact(str(rendered_video_path), artifact_path="video/postprocessed")
        mlflow.log_artifact(str(tracks_output_path), artifact_path="tracking/postprocessed")
        mlflow.log_artifact(str(info_output_path), artifact_path="tracking/postprocessed")
        mlflow.log_artifact(str(summary_output_path), artifact_path="tracking/postprocessed")
        mlflow.log_metrics({f"postprocessed_{key}": value for key, value in postprocessed_tracking_summary.items()})


if __name__ == "__main__":
    main()
