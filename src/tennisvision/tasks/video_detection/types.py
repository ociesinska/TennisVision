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

    # Number of frame results consumed, including frames without detections.
    processed_frame_count: int | None = None

    # Frame count reported by the source video; informational only.
    source_frame_count: int | None = None

    # Legacy lower bound, never an authoritative count of processed frames.
    # Persist it so removing detections cannot shorten the estimated timeline.
    inferred_frame_count: int | None = None

    def __post_init__(self) -> None:
        for name in ("processed_frame_count", "source_frame_count", "inferred_frame_count"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        for detection in self.detections:
            if type(detection.frame_id) is not int or detection.frame_id < 0:
                raise ValueError("frame_id must be a non-negative integer")
        observed_span = max((d.frame_id for d in self.detections), default=-1) + 1
        if self.processed_frame_count is not None:
            if observed_span > self.processed_frame_count:
                raise ValueError("Detection frame_id exceeds processed_frame_count")
        elif self.inferred_frame_count is None:
            self.inferred_frame_count = observed_span
        elif observed_span > self.inferred_frame_count:
            raise ValueError("Detection frame_id exceeds inferred_frame_count")

    @property
    def statistics_frame_count(self) -> int:
        """Denominator for this result; source metadata is never substituted."""
        if self.processed_frame_count is not None:
            return self.processed_frame_count
        return self.inferred_frame_count if self.inferred_frame_count is not None else 0

    @property
    def frame_count_basis(self) -> str:
        return "processed_frames" if self.processed_frame_count is not None else "detection_span_fallback"
