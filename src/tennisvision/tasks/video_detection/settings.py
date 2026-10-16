import os
from dataclasses import dataclass
from pathlib import Path

from tennisvision.core.settings import RuntimeSettings

DEFAULT_POSTPROCESSING_CONFIG_PATH = Path(__file__).parent / "configs" / "postprocessing" / "baseline_v1.yaml"


@dataclass(frozen=True)
class VideoDetectionSettings:
    runtime: RuntimeSettings

    model_uri: str | None
    run_id: str | None
    model_artifact_path: str
    model_path: Path

    imgsz: int
    confidence: float
    iou: float

    results_dir: Path
    postprocessing_config_path: Path
    default_max_players: int


def load_video_detection_settings(runtime: RuntimeSettings) -> VideoDetectionSettings:
    return VideoDetectionSettings(
        runtime=runtime,
        model_uri=os.getenv("VIDEO_DETECTION_MODEL_URI") or None,
        run_id=os.getenv("VIDEO_DETECTION_RUN_ID") or None,
        model_artifact_path=os.getenv(
            "VIDEO_DETECTION_MODEL_ARTIFACT_PATH",
            "weights/best.pt",
        ),
        model_path=Path(
            os.getenv(
                "VIDEO_DETECTION_MODEL_PATH",
                "data/artifacts/detection/yolo11s_baseline/weights/best.pt",
            )
        ),
        imgsz=int(os.getenv("VIDEO_DETECTION_IMGSZ", "1280")),
        confidence=float(os.getenv("VIDEO_DETECTION_CONFIDENCE", "0.5")),
        iou=float(os.getenv("VIDEO_DETECTION_IOU", "0.7")),
        results_dir=Path(
            os.getenv(
                "VIDEO_DETECTION_RESULTS_DIR",
                "data/artifacts/video_detection/api_results",
            )
        ),
        postprocessing_config_path=Path(
            os.getenv(
                "VIDEO_POSTPROCESSING_CONFIG_PATH",
                str(DEFAULT_POSTPROCESSING_CONFIG_PATH),
            )
        ),
        default_max_players=int(os.getenv("VIDEO_DETECTION_DEFAULT_MAX_PLAYERS", "2")),
    )
