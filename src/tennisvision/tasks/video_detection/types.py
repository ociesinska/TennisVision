from dataclasses import dataclass


@dataclass
class VideoFrameDetection:
    frame_id: int
    timestamp_sec: float
    class_id: int
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass
class VideoTrackDetection(VideoFrameDetection):
    track_id: int | None


@dataclass
class VideoTrackingResult:
    video_path: str
    width: int
    height: int
    fps: float
    detections: list[VideoTrackDetection]
