import os
import shutil
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from tennisvision.core.mlflow_utils import setup_mlflow
from tennisvision.tasks.detection.inference import get_model_source, load_detector
from tennisvision.tasks.video_detection.api.schemas import VideoTrackingResponse
from tennisvision.tasks.video_detection.postprocessing import TrackPostProcessingConfig, postprocess_tracking_result
from tennisvision.tasks.video_detection.tracking import VideoTrackingConfig, track_video
from tennisvision.tasks.video_detection.visualization import render_tracking_video


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")

    tracking_cfg = VideoTrackingConfig(
        model_path=Path(
            os.getenv(
                "VIDEO_DETECTION_MODEL_PATH",
                "data/artifacts/detection/yolo11s_baseline/weights/best.pt",
            )
        ),
        model_artifact_path=os.getenv("VIDEO_DETECTION_MODEL_ARTIFACT_PATH", "weights/best.pt"),
        imgsz=int(os.getenv("VIDEO_DETECTION_IMGSZ", "1280")),
        confidence=float(os.getenv("VIDEO_DETECTION_CONFIDENCE", "0.5")),
        iou=float(os.getenv("VIDEO_DETECTION_IOU", "0.7")),
        device=os.getenv("DEVICE", "auto"),
        run_id=os.getenv("VIDEO_DETECTION_RUN_ID") or None,
        model_uri=os.getenv("VIDEO_DETECTION_MODEL_URI") or None,
        tracking_uri=tracking_uri,
        save_video=False,
    )

    postprocessing_cfg = TrackPostProcessingConfig(
        min_count=int(os.getenv("VIDEO_POSTPROCESS_MIN_COUNT", "10")),
        min_presence_ratio=float(os.getenv("VIDEO_POSTPROCESS_MIN_PRESENCE_RATIO", "0.02")),
        min_mean_conf=float(os.getenv("VIDEO_POSTPROCESS_MIN_MEAN_CONF", "0.5")),
        min_mean_box_area_ratio=float(os.getenv("VIDEO_POSTPROCESS_MIN_MEAN_BOX_AREA_RATIO", "0.0002")),
        min_total_path_distance_ratio=float(os.getenv("VIDEO_POSTPROCESS_MIN_TOTAL_PATH_DISTANCE_RATIO", "0.005")),
        max_stitch_frame_gap=int(os.getenv("VIDEO_POSTPROCESS_MAX_STITCH_FRAME_GAP", "120")),
        max_stitch_gap_ratio=float(os.getenv("VIDEO_POSTPROCESS_MAX_STITCH_GAP_RATIO", "0.18")),
        max_stitch_overlap=int(os.getenv("VIDEO_POSTPROCESS_MAX_STITCH_OVERLAP", "10")),
        max_stitch_overlap_ratio=float(os.getenv("VIDEO_POSTPROCESS_MAX_STITCH_OVERLAP_RATIO", "0.01")),
        max_stitch_center_distance=float(os.getenv("VIDEO_POSTPROCESS_MAX_STITCH_CENTER_DISTANCE", "250")),
        max_stitch_center_distance_ratio=float(os.getenv("VIDEO_POSTPROCESS_MAX_STITCH_CENTER_DISTANCE_RATIO", "0.12")),
        max_stitch_area_ratio=float(os.getenv("VIDEO_POSTPROCESS_MAX_STITCH_AREA_RATIO", "2.0")),
        min_duplicate_overlap_frames=int(os.getenv("VIDEO_POSTPROCESS_MIN_DUPLICATE_OVERLAP_FRAMES", "5")),
        min_duplicate_iou=float(os.getenv("VIDEO_POSTPROCESS_MIN_DUPLICATE_IOU", "0.5")),
        max_tracks=int(os.getenv("VIDEO_POSTPROCESS_MAX_TRACKS")) if os.getenv("VIDEO_POSTPROCESS_MAX_TRACKS") else None,
        edge_margin_ratio=float(os.getenv("VIDEO_POSTPROCESS_EDGE_MARGIN_RATIO", "0.08")),
        max_reentry_frame_gap=int(os.getenv("VIDEO_POSTPROCESS_MAX_REENTRY_FRAME_GAP", "180")),
        max_reentry_gap_ratio=float(os.getenv("VIDEO_POSTPROCESS_MAX_REENTRY_GAP_RATIO", "0.25")),
        max_reentry_center_distance=float(os.getenv("VIDEO_POSTPROCESS_MAX_REENTRY_CENTER_DISTANCE", "700")),
        max_reentry_center_distance_ratio=float(os.getenv("VIDEO_POSTPROCESS_MAX_REENTRY_CENTER_DISTANCE_RATIO", "0.35")),
        max_reentry_area_ratio=float(os.getenv("VIDEO_POSTPROCESS_MAX_REENTRY_AREA_RATIO", "3.0")),
        reentry_side_ratio=float(os.getenv("VIDEO_POSTPROCESS_REENTRY_SIDE_RATIO", "0.4")),
    )

    setup_mlflow(experiment_name=None, tracking_uri=tracking_uri, set_experiment=False)

    model = load_detector(tracking_cfg)

    app.state.model = model
    app.state.tracking_cfg = tracking_cfg
    app.state.postprocessing_cfg = postprocessing_cfg
    app.state.device = tracking_cfg.device
    app.state.model_uri = get_model_source(tracking_cfg)

    yield

    del app.state.model


app = FastAPI(title="Player Video Tracking API", version="0.1.0", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "name": "Player Video Tracking API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health(request: Request):
    model = getattr(request.app.state, "model", None)
    device = getattr(request.app.state, "device", None)
    model_uri = getattr(request.app.state, "model_uri", None)

    return {"status": "ok" if model else "not_ready", "device": str(device), "model_uri": model_uri}


RESULTS_DIR = Path("data/artifacts/video_detection/api_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/track_postprocess", response_model=VideoTrackingResponse)
async def track_postprocess(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    max_players: int | None = Form(default=None, ge=1, le=4),  # noqa: B008
):
    tracking_cfg = request.app.state.tracking_cfg
    postprocessing_cfg = request.app.state.postprocessing_cfg
    request_postprocessing_cfg = replace(
        postprocessing_cfg,
        max_tracks=max_players if max_players is not None else postprocessing_cfg.max_tracks,
    )

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_path: Path | None = None
    output_path: Path | None = None

    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp, length=1024 * 1024)
            tmp_path = Path(tmp.name)

        model = request.app.state.model

        raw_result = track_video(
            cfg=tracking_cfg,
            video_path=tmp_path,
            model=model,
        )
        processed_result, _ = postprocess_tracking_result(raw_result, request_postprocessing_cfg)

        result_id = uuid4().hex
        output_path = RESULTS_DIR / f"{result_id}.mp4"

        render_tracking_video(video_path=tmp_path, tracking_result=processed_result, output_path=output_path)

        video_url = str(request.url_for("get_result_video", result_id=result_id))

        return VideoTrackingResponse.from_result(
            processed_result,
            video_url=video_url,
            result_id=result_id,
        )
    except Exception:
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        raise

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@app.get("/results/{result_id}/video.mp4", name="get_result_video", response_class=FileResponse)
def get_result_video(result_id: str) -> FileResponse:
    try:
        normalized_id = UUID(result_id).hex
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Video not found.") from exc

    video_path = RESULTS_DIR / f"{normalized_id}.mp4"

    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found.")

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="{normalized_id}.mp4"'},
    )
