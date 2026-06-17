import logging
import os
from contextlib import asynccontextmanager
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tennisvision.core.mlflow_utils import setup_mlflow
from tennisvision.core.utils import get_device
from tennisvision.tasks.detection.inference import (
    DetectionInferenceConfig,
    get_model_source,
    load_detector,
    predict_image,
)
from tennisvision.tasks.detection.visualization import viz_detected_boxes

logger = logging.getLogger(__name__)


class BoundingBoxResponse(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionItemResponse(BaseModel):
    class_id: int
    label: str
    confidence: float | None = 0.25
    box: BoundingBoxResponse


class DetectionResponse(BaseModel):
    width: int
    height: int
    detections: list[DetectionItemResponse]
    model_uri: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")

    cfg = DetectionInferenceConfig(
        backend=os.getenv("DETECTION_BACKEND", "ultralytics"),
        model_path=Path(
            os.getenv(
                "DETECTION_MODEL_PATH",
                "data/artifacts/detection/yolo11n_baseline/weights/best.pt",
            )
        ),
        model_artifact_path=os.getenv("DETECTION_MODEL_ARTIFACT_PATH", "models/best.pt"),
        imgsz=int(os.getenv("DETECTION_IMGSZ", "960")),
        confidence=float(os.getenv("DETECTION_CONFIDENCE", "0.25")),
        iou=float(os.getenv("DETECTION_IOU", "0.7")),
        device=os.getenv("DEVICE", "auto"),
        visualize=True,
        model_uri=os.getenv("DETECTION_MODEL_URI") or None,
        run_id=os.getenv("DETECTION_RUN_ID") or None,
        tracking_uri=tracking_uri,
    )

    setup_mlflow(experiment_name=None, tracking_uri=tracking_uri, set_experiment=False)

    model = load_detector(cfg)

    app.state.model = model
    app.state.cfg = cfg
    app.state.device = get_device(cfg.device)
    app.state.model_uri = get_model_source(cfg)

    yield

    del app.state.model


app = FastAPI(title="Player Detection API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    model = getattr(request.app.state, "model", None)
    device = getattr(request.app.state, "device", None)
    model_uri = getattr(request.app.state, "model_uri", None)

    return {
        "status": "ok" if model else "not_ready",
        "device": str(device),
        "model_uri": model_uri,
    }


@app.post("/predict", response_model=DetectionResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    confidence: float | None = None,
    iou: float | None = None,
):
    cfg = request.app.state.cfg
    request_cfg = replace(
        cfg,
        confidence=confidence if confidence is not None else cfg.confidence,
        iou=iou if iou is not None else cfg.iou,
    )
    model = request.app.state.model
    tmp_path: Path | None = None

    try:
        suffix = Path(file.filename or "image.jpg").suffix or ".jpg"

        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        result = predict_image(
            model=model,
            image_path=tmp_path,
            cfg=request_cfg,
        )

        return DetectionResponse(
            width=result.width,
            height=result.height,
            detections=[
                DetectionItemResponse(
                    class_id=d.class_id,
                    label=d.label,
                    confidence=d.confidence,
                    box=BoundingBoxResponse(
                        x1=d.xyxy[0],
                        y1=d.xyxy[1],
                        x2=d.xyxy[2],
                        y2=d.xyxy[3],
                    ),
                )
                for d in result.detections
            ],
            model_uri=request.app.state.model_uri,
        )

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@app.post("/predict/image", response_class=StreamingResponse)
async def predict_image_with_boxes(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    confidence: float | None = None,
    iou: float | None = None,
) -> StreamingResponse:
    cfg = request.app.state.cfg
    request_cfg = replace(
        cfg,
        confidence=confidence if confidence is not None else cfg.confidence,
        iou=iou if iou is not None else cfg.iou,
    )
    model = request.app.state.model
    tmp_path: Path | None = None

    try:
        suffix = Path(file.filename or "image.jpg").suffix or ".jpg"

        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        result = predict_image(
            model=model,
            image_path=tmp_path,
            cfg=request_cfg,
        )

        image = viz_detected_boxes(result)

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return StreamingResponse(
        content=buffer,
        media_type="image/png",
        headers={"Content-Disposition": 'inline; filename="detections.png"'},
    )
