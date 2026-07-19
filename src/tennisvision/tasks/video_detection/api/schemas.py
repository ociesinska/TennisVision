from pydantic import BaseModel

from tennisvision.tasks.video_detection.types import VideoTrackingResult


class VideoTrackDetectionResponse(BaseModel):
    frame_id: int
    timestamp_sec: float
    class_id: int
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None


class VideoTrackingResponse(BaseModel):
    video_url: str
    result_id: str
    width: int
    height: int
    fps: float
    detections: list[VideoTrackDetectionResponse]

    @classmethod
    def from_result(cls, result: VideoTrackingResult, video_url: str, result_id: str) -> "VideoTrackingResponse":
        return cls(
            video_url=video_url,
            result_id=result_id,
            width=result.width,
            height=result.height,
            fps=result.fps,
            detections=[
                VideoTrackDetectionResponse(
                    frame_id=d.frame_id,
                    timestamp_sec=d.timestamp_sec,
                    class_id=d.class_id,
                    label=d.label,
                    confidence=d.confidence,
                    xyxy=d.xyxy,
                    track_id=d.track_id,
                )
                for d in result.detections
            ],
        )
