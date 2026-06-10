from dataclasses import dataclass


@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass
class DetectionResult:
    image_path: str | None
    width: int
    height: int
    detections: list[Detection]