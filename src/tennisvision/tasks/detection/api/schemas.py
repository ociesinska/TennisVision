from pydantic import BaseModel


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
