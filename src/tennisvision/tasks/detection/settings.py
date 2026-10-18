import os
from dataclasses import dataclass
from pathlib import Path

from tennisvision.core.settings import RuntimeSettings


@dataclass(frozen=True)
class DetectionSettings:
    runtime: RuntimeSettings

    backend: str
    model_uri: str
    run_id: str | None
    model_artifact_path: str
    model_path: Path

    imgsz: int
    confidence: float
    iou: float


def load_detection_settings(runtime: RuntimeSettings) -> DetectionSettings:
    return DetectionSettings(
        runtime=runtime,
        backend=os.getenv(
            "DETECTION_BACKEND",
            "ultralytics",
        ),
        model_uri=os.getenv("DETECTION_MODEL_URI") or None,
        run_id=os.getenv("DETECTION_RUN_ID") or None,
        model_artifact_path=os.getenv(
            "DETECTION_MODEL_ARTIFACT_PATH",
            "weights/best.pt",
        ),
        model_path=Path(
            os.getenv(
                "DETECTION_MODEL_PATH",
                "data/artifacts/detection/yolo11n_baseline/weights/best.pt",
            )
        ),
        imgsz=int(os.getenv("DETECTION_IMGSZ", "960")),
        confidence=float(os.getenv("DETECTION_CONFIDENCE", "0.25")),
        iou=float(os.getenv("DETECTION_IOU", "0.7")),
    )
