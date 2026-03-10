from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
from pydantic import BaseModel

from tennisvision.core.data import build_preprocess
from tennisvision.core.explainability import gradcam_heatmap, overlay_heatmap, preprocess_PIL
from tennisvision.core.mlflow_utils import load_model_from_mlflow, setup_mlflow
from tennisvision.core.utils import concat_rgb, rgb_ndarray_to_png_bytes

logger = logging.getLogger()

PREPROCESS = build_preprocess()

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
    setup_mlflow(experiment_name=None, tracking_uri=tracking_uri, set_experiment=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # loading model from the registry
    model_uri_env = os.getenv("MODEL_URI", "models:/TennisVision@champion")
    # alternatively loading model from a specific run
    run_id = os.getenv("RUN_ID", "")

    if model_uri_env:
        model, resolved_uri = load_model_from_mlflow(model_uri=model_uri_env, device=device)
    elif run_id:
        model, resolved_uri = load_model_from_mlflow(run_id=run_id, device=device)
    else:
        raise RuntimeError("Set MODEL_URI or RUN_ID env var for the API.")

    model.eval()

    app.state.model = model
    app.state.device = device
    app.state.model_uri = resolved_uri
    app.state.idx_to_class = {0: "backhand", 1: "forehand", 2: "ready_position", 3: "serve"}

    yield

    # shutdown: cleanup
    del app.state.model


app = FastAPI(title="Shot Classification API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    model = getattr(request.app.state, "model", None)
    device = getattr(request.app.state, "device", None)
    model_uri = getattr(request.app.state, "model_uri", None)

    return {"status": "ok" if model else "not_ready", "device": str(device), "model_uri": model_uri}


@torch.inference_mode()
def _predict_pil(request: Request, img: Image.Image, top_k: int = 3) -> tuple[int, float, list[tuple[int, float]]]:

    model = request.app.state.model
    device = request.app.state.device

    assert model is not None

    img = img.convert("RGB")
    x = PREPROCESS(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu()

    conf, pred_idx = torch.max(probs, dim=0)
    top_probs, top_idx = torch.topk(probs, k=min(top_k, probs.numel()))

    top = list(zip(top_idx.tolist(), top_probs.tolist(), strict=True))

    return int(pred_idx.item()), float(conf.item()), top


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, file: UploadFile = File(...), top_k: int = 3):
    try:
        model = request.app.state.model
        idx_to_class = request.app.state.idx_to_class
        model_uri = request.app.state.model_uri

        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet.")
        if not (file.content_type or "").startswith("image/"):
            raise HTTPException(status_code=400, detail=f"Unsupported content_type: {file.content_type}")

        raw = await file.read()

        if len(raw) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=413, detail="File too large (>10MB).")

        try:
            img = Image.open(io.BytesIO(raw))
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image file.") from e

        t0 = time.time()
        pred_idx, conf, top = _predict_pil(request, img, top_k=top_k)
        dt = time.time() - t0

        label = idx_to_class.get(pred_idx, str(pred_idx))
        topk_payload = [{"label": idx_to_class.get(i, str(i)), "p": float(p)} for i, p in top]

        resp = PredictResponse(label=label, confidence=conf, topk=topk_payload, model_uri=model_uri)

        return JSONResponse(content={**resp.model_dump(), "latency_ms": round(dt * 1000, 2)})
    except Exception as e:
        logger.exception("Predict failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@torch.inference_mode()
def _predict_batch_pil(request: Request, images: list[Image.Image], top_k: int) -> tuple[list[int], list[float], list[list[tuple[int, float]]]]:

    model = request.app.state.model
    device = request.app.state.device

    x = torch.stack([PREPROCESS(img.convert("RGB")) for img in images], dim=0).to(device)  # [B, C, H, W]
    logits = model(x)  # [B, num_classes]
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
async def predict_batch(request: Request, files: list[UploadFile] = File(...), top_k: int = 3, strict: bool = False):
    """
    strict = False: returns error per file and moves on
    strict = True: wrong file -> error
    """
    try:
        model = request.app.state.model
        device = request.app.state.device
        idx_to_class = request.app.state.idx_to_class
        model_uri = request.app.state.model_uri

        if files is None:
            files = File(...)
        if model is None:
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
            return BatchPredictResponse(model_uri=model_uri, device=str(device), latency_ms=0.0, results=results)

        t0 = time.time()
        pred_idx, conf, top = _predict_batch_pil(request=request, images=images, top_k=top_k)
        dt_ms = (time.time() - t0) * 1000

        for fname, pi, ci, top_i in zip(filenames, pred_idx, conf, top, strict=True):
            label = idx_to_class.get(pi, str(pi))
            topk_payload = [TopKItem(label=idx_to_class.get(i, str(i)), p=float(p)) for i, p in top_i]
            results.append(BatchItem(filename=fname, label=label, confidence=float(ci), topk=topk_payload))

        return BatchPredictResponse(model_uri=model_uri, device=str(device), latency_ms=round(dt_ms, 2), results=results)

    except Exception as e:
        logger.exception("Predict Batch failed.")
        raise HTTPException(status_code=500, detail=str(e)) from e
    

@app.post("/explain") #TODO make it predict explain
async def explain(request: Request, file: UploadFile = File(...)):
    raw = await file.read()

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported content_type: {file.content_type}")

    image = Image.open(io.BytesIO(raw))
    model = request.app.state.model
    device = request.app.state.device
    request.app.state.idx_to_class = {0: "backhand", 1: "forehand", 2: "ready_position", 3: "serve"}
    idx_to_class = app.state.idx_to_class # TODO

    x = preprocess_PIL(image, PREPROCESS)
    x = x.to(device)
    pred = model(x) # [B, C]
    top2_idx = pred.topk(k=2, dim=1).indices # [B, 2]
    pred_idx1 = top2_idx[:, 0].item()
    pred_idx2 = top2_idx[:, 1].item()
    pred_class_1 = idx_to_class[pred_idx1]
    pred_class_2 = idx_to_class[pred_idx2]
    conv_layer = model.features[-2] #TODO adjust to other model types

    heatmap_pred1 = gradcam_heatmap(model=model, x=x, target=pred_idx1, conv_layer=conv_layer, device=device)
    heatmap_pred2 = gradcam_heatmap(model=model, x=x, target=pred_idx2, conv_layer=conv_layer, device=device)

    overlay_pred1 = overlay_heatmap(image, heatmap_pred1, alpha=0.4, is_rgb=True)
    overlay_pred2 = overlay_heatmap(image, heatmap_pred2, alpha=0.4, is_rgb=True)

    combined = concat_rgb(overlay_pred1, overlay_pred2)
    png_bytes = rgb_ndarray_to_png_bytes(combined)

    return Response(content=png_bytes, media_type="image/png")

    