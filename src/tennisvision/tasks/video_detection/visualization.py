from pathlib import Path

import cv2

from tennisvision.tasks.video_detection.types import VideoTrackingResult

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def render_tracking_video(video_path: Path, tracking_result: VideoTrackingResult, output_path: Path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise ValueError(f"Could not open video: {video_path}")

    fps = tracking_result.fps
    width, height = tracking_result.width, tracking_result.height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Could not open video writer: {output_path}")

    detections_by_frame = {}
    for detection in tracking_result.detections:
        detections_by_frame.setdefault(detection.frame_id, []).append(detection)

    for frame_id in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break

        frame_detections = detections_by_frame.get(frame_id, [])

        for detection in frame_detections:
            x1, y1, x2, y2 = map(int, detection.xyxy)

            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 4)

            label = f"id:{detection.track_id} {detection.label} {detection.confidence:.2f}"
            font_scale = 1.4
            thickness = 3
            padding = 6
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            label_y1 = max(y1 - text_height - baseline - (2 * padding), 0)
            label_y2 = label_y1 + text_height + baseline + (2 * padding)
            label_x2 = min(x1 + text_width + (2 * padding), width)

            cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), BOX_COLOR, -1)
            cv2.putText(
                frame,
                label,
                (x1 + padding, label_y2 - baseline - padding),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                TEXT_COLOR,
                thickness,
            )

        writer.write(frame)

    cap.release()
    writer.release()

    return output_path
