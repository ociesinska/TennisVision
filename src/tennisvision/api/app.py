from __future__ import annotations

import io
import logging
import os
import time

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms as T

from tennisvision.core.mlflow_utils import load_model_from_mlflow, setup_mlflow

logger = logging.getLogger()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PREPROCESS = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class PredictResponse(BaseModel):
    label: str
    confidence: float
    topk: list[dict]
    model_uri: str


class TopKItem(BaseModel):
    label: str
    p: float


class BatchItem(BaseModel):
    filename: str
    label: str | None = None
    confidence: float | None = None
    topk: list[TopKItem] | None = None
    error: str | None = None


class BatchPredictResponse(BaseModel):
    model_uri: str
    device: str
    latency_ms: float
    results: list[BatchItem]


app = FastAPI(title="Shot Classification API", version="0.1.0")

MODEL: torch.nn.Module | None = None
IDX_TO_CLASS: dict[int, str] = {}
DEVICE: torch.device
MODEL_URI: str = "models:/TennisVision@champion"


@app.on_event("startup")
def _startup() -> None:
    global MODEL, IDX_TO_CLASS, DEVICE, MODEL_URI

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
    setup_mlflow(experiment_name=None, tracking_uri=tracking_uri, set_experiment=False)

    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # loading model from the registry
    MODEL_URI = os.getenv("MODEL_URI", MODEL_URI)
    # alternatively loading model from a specific run
    RUN_ID = os.getenv("RUN_ID", "")

    if MODEL_URI:
        model, uri = load_model_from_mlflow(model_uri=MODEL_URI, device=DEVICE)
    elif RUN_ID:
        model, uri = load_model_from_mlflow(run_id=RUN_ID, device=DEVICE)
    else:
        raise RuntimeError("Set MODEL_URI or MODEL_RUN_ID env var for the API.")

    model.eval()
    MODEL = model

    IDX_TO_CLASS = {0: "backhand", 1: "forehand", 2: "ready_position", 3: "serve"}


@app.get("/health")
def health():
    ok = MODEL is not None
    return {"status": "ok" if ok else "not_ready", "device": str(DEVICE), "model_uri": MODEL_URI}


@torch.inference_mode()
def _predict_pil(img: Image.Image, top_k: int = 3) -> tuple[int, float, list[tuple[int, float]]]:
    assert MODEL is not None

    img = img.convert("RGB")
    x = PREPROCESS(img).unsqueeze(0).to(DEVICE)

    logits = MODEL(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu()

    conf, pred_idx = torch.max(probs, dim=0)
    top_probs, top_idx = torch.topk(probs, k=min(top_k, probs.numel()))

    top = list(zip(top_idx.tolist(), top_probs.tolist(), strict=True))

    return int(pred_idx.item()), float(conf.item()), top


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = None, top_k: int = 3):
    try:
        if MODEL is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet.")
        if not (file.content_type or "").startswith("image/"):
            raise HTTPException(status_code=400, detail=f"Unsupported content_type: {file.content_type}")
        if file is None:
            file = File(...)
        raw = await file.read()
        if len(raw) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=413, detail="File too large (>10MB).")

        try:
            img = Image.open(io.BytesIO(raw))
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image file.") from e

        t0 = time.time()
        pred_idx, conf, top = _predict_pil(img, top_k=top_k)
        dt = time.time() - t0

        label = IDX_TO_CLASS.get(pred_idx, str(pred_idx))
        topk_payload = [{"label": IDX_TO_CLASS.get(i, str(i)), "p": float(p)} for i, p in top]

        resp = PredictResponse(label=label, confidence=conf, topk=topk_payload, model_uri=MODEL_URI)

        return JSONResponse(content={**resp.model_dump(), "latency_ms": round(dt * 1000, 2)})
    except Exception as e:
        logger.exception("Predict failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@torch.inference_mode()
def _predict_batch_pil(images: list[Image.Image], top_k: int) -> tuple[list[int], list[float], list[list[tuple[int, float]]]]:
    x = torch.stack([PREPROCESS(img.convert("RGB")) for img in images], dim=0).to(DEVICE)  # [B, C, H, W]
    logits = MODEL(x)  # [B, num_classes]
    probs = torch.softmax(logits, dim=1)  # [B, num_classes]

    top_k = min(top_k, probs.size(1))
    top_probs, top_idx = torch.topk(probs, k=top_k, dim=1)  # [B, K], [B, K]

    pred_idx = probs.argmax(dim=1)  # [B]
    conf = probs.gather(1, pred_idx.unsqueeze(1)).squeeze(1)  # [B]

    pred_idx_list_cpu = pred_idx.detach().cpu().tolist()
    conf_list_cpu = conf.detach().cpu().tolist()

    top_prob_idx_cpu = top_idx.detach().cpu().tolist()
    top_probs_cpu = top_probs.detach().cpu().tolist()

    # Iterate over images, not top_k!
    top = []
    for i in range(len(images)):
        top.append(list(zip(top_prob_idx_cpu[i], top_probs_cpu[i], strict=True)))

    return pred_idx_list_cpu, conf_list_cpu, top


@app.post("/predict_batch", response_model=BatchPredictResponse)
async def predict_batch(files: list[UploadFile] = None, top_k: int = 3, strict: bool = False):
    """
    strict = False: returns error per file and moves on
    strict = True: wrong file -> error
    """

    if files is None:
        files = File(...)
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > 64:
        raise HTTPException(status_code=413, detail="Too many files (max 64).")

    images: list[Image.Image] = []
    filenames: list[str] = []
    results: list[BatchItem] = []

    total_bytes = 0
    for f in files:
        if not (f.content_type or "").startswith("image/"):
            msg = f"Unsupported content type {f.content_type}"
            if strict:
                raise HTTPException(status_code=400, detail=msg)
            results.append(BatchItem(filename=f.filename, error=msg))
            continue

        raw = await f.read()
        total_bytes += len(raw)
        if total_bytes > 25 * 1024 * 1024:  # 25 MB limit per the whole request
            raise HTTPException(status_code=413, detail="Batch payload too lage (>25MB).")

        try:
            img = Image.open(io.BytesIO(raw))
        except Exception as e:
            msg = "Invalid image file."
            if strict:
                raise HTTPException(status_code=400, detail=msg) from e
            results.append(BatchItem(filename=f.filename, error=msg))
            continue

        images.append(img)
        filenames.append(f.filename)

    # all invalid images
    if not images:
        return BatchPredictResponse(model_uri=MODEL_URI, device=str(DEVICE), latency_ms=0.0, results=results)

    t0 = time.time()
    pred_idx, conf, top = _predict_batch_pil(images=images, top_k=top_k)
    dt_ms = (time.time() - t0) * 100

    for fname, pi, ci, top_i in zip(filenames, pred_idx, conf, top, strict=True):
        label = IDX_TO_CLASS.get(pi, str(pi))
        topk_payload = [TopKItem(label=IDX_TO_CLASS.get(i, str(i)), p=float(p)) for i, p in top_i]
        results.append(BatchItem(filename=fname, label=label, confidence=float(ci), topk=topk_payload))

    return BatchPredictResponse(model_uri=MODEL_URI, device=str(DEVICE), latency_ms=round(dt_ms, 2), results=results)
