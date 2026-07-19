from pathlib import Path
from typing import Any

import cv2

from tennisvision.core.utils import get_device
from tennisvision.tasks.video_detection.types import VideoTrackDetection, VideoTrackingResult


def read_video_metadata(video_path: str | Path) -> tuple[int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    cap.release()

    return width, height, fps


def track_ultralytics_video(
    model: Any,
    video_path: str | Path,
    cfg,
) -> VideoTrackingResult:

    device = get_device(cfg.device)

    results = model.track(
        source=str(video_path),
        tracker=cfg.tracker,
        conf=cfg.confidence,
        iou=cfg.iou,
        imgsz=cfg.imgsz,
        device=device,
        persist=True,
        stream=True,
        save=cfg.save_video,
        project=str(cfg.output_dir),
        name=cfg.run_name,
        verbose=False,
    )

    detections = []
    width, height, fps = read_video_metadata(video_path)

    for frame_id, result in enumerate(results):
        timestamp = frame_id / fps if fps > 0 else 0.0

        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            track_id = None

            if box.id is not None:
                track_id = int(box.id.item())

            detections.append(
                VideoTrackDetection(
                    frame_id=frame_id,
                    timestamp_sec=timestamp,
                    track_id=track_id,
                    class_id=class_id,
                    label=result.names[class_id],
                    confidence=confidence,
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                )
            )

    return VideoTrackingResult(
        video_path=str(video_path),
        width=int(width),
        height=int(height),
        fps=fps,
        detections=detections,
    )
