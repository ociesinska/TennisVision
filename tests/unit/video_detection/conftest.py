import pytest

from tennisvision.tasks.video_detection.types import VideoTrackDetection, VideoTrackingResult


@pytest.fixture
def detection():
    """Build explicit synthetic detections without videos or model dependencies."""

    def make(frame=0, track=1, box=(40.0, 40.0, 50.0, 60.0), confidence=0.8):
        return VideoTrackDetection(
            frame_id=frame,
            timestamp_sec=frame / 25,
            class_id=0,
            label="player",
            confidence=confidence,
            xyxy=box,
            track_id=track,
        )

    return make


@pytest.fixture
def tracking_result():
    def make(detections=(), width=100, height=100):
        return VideoTrackingResult("synthetic.mp4", width, height, 25.0, list(detections))

    return make
