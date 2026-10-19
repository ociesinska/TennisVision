from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrackingObservation:
    frame_id: int
    track_id: int | None
    xyxy: tuple[float, float, float, float]
    confidence: float | None = None
    outside: bool | None = None
    occluded: bool | None = None
    keyframe: bool | None = None


@dataclass(frozen=True)
class TrackingSequence:
    sequence_id: str
    frame_count: int
    width: int
    height: int
    fps: float | None
    track_labels: dict[int, str]
    observations: tuple[TrackingObservation, ...]
    source_path: Path
    source_format: str
    warnings: tuple[str, ...] = ()
