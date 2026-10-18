from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tennisvision.tasks.video_detection.backends import ultralytics_video as backend


@pytest.fixture
def tracking_config():
    return SimpleNamespace(
        device="cpu",
        tracker="bytetrack.yaml",
        confidence=0.25,
        iou=0.7,
        imgsz=960,
        save_video=False,
        output_dir=Path("unused-output"),
        run_name="test",
    )


@pytest.mark.parametrize(
    "frame_kinds,expected_count,expected_detection_frames",
    [
        ([], 0, []),
        (["none", "empty", "none"], 3, []),
        (["detected", "none", "empty"], 3, [0]),
        (["empty", "detected", "none"], 3, [1]),
    ],
    ids=["empty-stream", "no-detections", "trailing-empty-frames", "leading-and-trailing-empty-frames"],
)
@pytest.mark.parametrize("source_count", [100, None], ids=["reported-source-count", "unknown-source-count"])
def test_counts_processed_frames_independently_of_detections(
    monkeypatch,
    tracking_config,
    frame_kinds,
    expected_count,
    expected_detection_frames,
    source_count,
):
    # Mimic only the tensor methods consumed by the adapter; no inference runs.
    box = SimpleNamespace(
        cls=Mock(item=Mock(return_value=0)),
        conf=Mock(item=Mock(return_value=0.8)),
        id=Mock(item=Mock(return_value=7)),
        xyxy=[Mock(tolist=Mock(return_value=[10, 20, 30, 40]))],
    )
    boxes = {"none": None, "empty": [], "detected": [box]}
    results = (SimpleNamespace(boxes=boxes[kind], names={0: "player"}) for kind in frame_kinds)
    model = Mock()
    model.track.return_value = results
    metadata_reader = Mock(return_value=(640, 480, 25.0, source_count))
    monkeypatch.setattr(backend, "read_video_metadata", metadata_reader)

    result = backend.track_ultralytics_video(model, Path("synthetic.mp4"), tracking_config)

    assert result.processed_frame_count == expected_count
    assert result.source_frame_count == source_count
    assert (result.video_path, result.width, result.height, result.fps) == ("synthetic.mp4", 640, 480, 25.0)
    assert [d.frame_id for d in result.detections] == expected_detection_frames
    assert [d.timestamp_sec for d in result.detections] == pytest.approx([f / 25 for f in expected_detection_frames])
    for detection in result.detections:
        assert detection.track_id == 7
        assert detection.xyxy == (10.0, 20.0, 30.0, 40.0)
    metadata_reader.assert_called_once_with(Path("synthetic.mp4"))
    model.track.assert_called_once()
    assert model.track.call_args.kwargs["device"] == "cpu"
    assert model.track.call_args.kwargs["stream"] is True


@pytest.fixture
def capture(monkeypatch):
    capture = Mock()
    capture.isOpened.return_value = True
    monkeypatch.setattr(backend.cv2, "VideoCapture", Mock(return_value=capture))
    return capture


@pytest.mark.parametrize(
    "reported_count,expected",
    [(100.0, 100), (0.0, None), (-1.0, None), (float("nan"), None), (float("inf"), None), (float("-inf"), None)],
    ids=["valid", "unavailable", "negative", "nan", "positive-infinity", "negative-infinity"],
)
def test_reads_source_metadata_and_releases_capture(capture, reported_count, expected):
    properties = {
        backend.cv2.CAP_PROP_FRAME_WIDTH: 640,
        backend.cv2.CAP_PROP_FRAME_HEIGHT: 480,
        backend.cv2.CAP_PROP_FPS: 25.0,
        backend.cv2.CAP_PROP_FRAME_COUNT: reported_count,
    }
    capture.get.side_effect = properties.__getitem__

    assert backend.read_video_metadata(Path("synthetic.mp4")) == (640, 480, 25.0, expected)
    capture.release.assert_called_once_with()


def test_releases_capture_when_video_cannot_be_opened(capture):
    capture.isOpened.return_value = False
    with pytest.raises(ValueError, match="Could not open video missing.mp4"):
        backend.read_video_metadata("missing.mp4")
    capture.get.assert_not_called()
    capture.release.assert_called_once_with()


def test_releases_capture_when_metadata_read_fails(capture):
    capture.get.side_effect = RuntimeError("metadata read failed")
    with pytest.raises(RuntimeError, match="metadata read failed"):
        backend.read_video_metadata("synthetic.mp4")
    capture.release.assert_called_once_with()
