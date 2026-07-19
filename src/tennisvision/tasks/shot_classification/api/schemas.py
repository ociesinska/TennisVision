from pydantic import BaseModel


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
