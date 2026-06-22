from dataclasses import dataclass
from pathlib import Path

import cv2

from tennisvision.tasks.detection.inference import load_detector
from tennisvision.tasks.video_detection.backends.ultralytics_video import track_ultralytics_video
from tennisvision.tasks.video_detection.types import VideoTrackingResult


@dataclass
class VideoTrackingConfig:
    backend: str = "ultralytics"
    model_path: Path = Path("data/artifacts/detection/yolo11s_baseline/weights/best.pt")
    model_uri: str | None = None
    run_id: str | None = None
    model_artifact_path: str = "weights/best.pt"
    tracking_uri: str | None = None

    imgsz: int = 960
    confidence: float = 0.25
    iou: float = 0.7
    device: str = "auto"

    tracker: str = "bytetrack.yaml"
    save_video: bool = True
    output_dir: Path = Path("data/artifacts/video_detection")
    run_name: str = "video_tracking"


def track_video(cfg: VideoTrackingConfig, video_path: str | Path) -> VideoTrackingResult:
    model = load_detector(cfg)

    if cfg.backend == "ultralytics":
        result = track_ultralytics_video(model, video_path, cfg)
        return result

    raise ValueError(f"Unknown backend: {cfg.backend}")


def save_sample_frames(video_path: Path, output_dir: Path) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return []

    sample_dir = output_dir / "sample_frames"
    sample_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = {
        "first": 0,
        "quarter": frame_count // 4,
        "middle": frame_count // 2,
        "three_quarter": (3 * frame_count) // 4,
        "last": max(frame_count - 1, 0),
    }

    saved_paths = []

    for name, frame_idx in frame_indices.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        frame_path = sample_dir / f"frame_{name}.jpg"
        cv2.imwrite(str(frame_path), frame)
        saved_paths.append(frame_path)

    cap.release()
    return saved_paths


def compute_tracking_summary(result: VideoTrackingResult) -> dict[str, float | int]:

    detections_per_frame: dict[int, int] = {}

    for detection in result.detections:
        detections_per_frame[detection.frame_id] = detections_per_frame.get(detection.frame_id, 0) + 1

    if not detections_per_frame:
        return {
            "min_detections_per_frame": 0,
            "max_detections_per_frame": 0,
            "mean_detections_per_frame": 0.0,
            "frames_with_detections": 0,
            "frames_with_detections_ratio": 0.0,
            "unique_track_ids": 0,
        }

    counts = list(detections_per_frame.values())

    unique_track_ids = {detection.track_id for detection in result.detections if detection.track_id is not None}

    total_frames = max(detections_per_frame) + 1

    return {
        "min_detections_per_frame": min(counts),
        "max_detections_per_frame": max(counts),
        "mean_detections_per_frame": sum(counts) / total_frames,
        "frames_with_detections": len(detections_per_frame),
        "frames_with_detections_ratio": len(detections_per_frame) / total_frames,
        "unique_track_ids": len(unique_track_ids),
    }
