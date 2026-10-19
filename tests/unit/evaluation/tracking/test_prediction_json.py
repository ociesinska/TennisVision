import json

import pytest

from tennisvision.evaluation.tracking.io import load_prediction_json
from tennisvision.evaluation.tracking.types import TrackingObservation


@pytest.fixture
def prediction_file(tmp_path):
    data = {
        "video_path": "video/example.mp4",
        "width": 100,
        "height": 80,
        "fps": 25.0,
        "processed_frame_count": 10,
        "source_frame_count": 100,
        "inferred_frame_count": 2,
        "detections": [{"frame_id": 1, "track_id": 7, "xyxy": [1.5, 2, 10, 20], "confidence": 0.875, "label": "player"}],
    }

    def save():
        path = tmp_path / "tracks.json"
        path.write_text(json.dumps(data))
        return path

    return data, save


def test_preserves_observation_and_authoritative_timeline(prediction_file):
    _, save = prediction_file
    path = save()
    result = load_prediction_json(path)
    assert (result.sequence_id, result.frame_count, result.width, result.height, result.fps) == ("example", 10, 100, 80, 25.0)
    assert result.observations == (TrackingObservation(1, 7, (1.5, 2, 10, 20), confidence=0.875),)
    assert result.track_labels == {7: "player"}
    assert result.source_path == path
    assert result.source_format == "tennisvision_prediction_json"
    assert result.warnings == ()
    assert set(range(result.frame_count)) - {o.frame_id for o in result.observations} == {0, 2, 3, 4, 5, 6, 7, 8, 9}


@pytest.mark.parametrize("count", [0, 10])
def test_empty_predictions_preserve_timeline(prediction_file, count):
    data, save = prediction_file
    data.update(detections=[], processed_frame_count=count)
    result = load_prediction_json(save())
    assert result.observations == ()
    assert result.track_labels == {}
    assert result.frame_count == count


@pytest.mark.parametrize("key", ["video_path", "width", "height", "fps", "processed_frame_count", "detections"])
def test_missing_metadata_is_rejected(prediction_file, key):
    data, save = prediction_file
    del data[key]
    with pytest.raises(ValueError, match=key):
        load_prediction_json(save())


@pytest.mark.parametrize(
    "key,value",
    [
        ("processed_frame_count", None),
        ("processed_frame_count", -1),
        ("processed_frame_count", True),
        ("processed_frame_count", 10.0),
        ("processed_frame_count", "10"),
        ("width", 0),
        ("height", -1),
        ("width", False),
        ("fps", 0),
        ("fps", -1),
        ("fps", float("nan")),
        ("fps", float("inf")),
        ("fps", "25"),
        ("video_path", ""),
        ("video_path", None),
        ("detections", {}),
    ],
)
def test_invalid_metadata_is_rejected(prediction_file, key, value):
    data, save = prediction_file
    data[key] = value
    with pytest.raises(ValueError, match=key):
        load_prediction_json(save())


@pytest.mark.parametrize(
    "key,value",
    [
        ("frame_id", -1),
        ("frame_id", 10),
        ("frame_id", 1.5),
        ("frame_id", True),
        ("track_id", None),
        ("track_id", -1),
        ("track_id", "7"),
        ("track_id", False),
        ("confidence", None),
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("confidence", float("nan")),
        ("confidence", float("inf")),
        ("confidence", True),
        ("xyxy", [1, 2, 1, 20]),
        ("xyxy", [1, 20, 10, 2]),
        ("xyxy", [1, 2, 3]),
        ("xyxy", [1, 2, float("inf"), 20]),
        ("xyxy", [None, 2, 10, 20]),
        ("xyxy", ["1", 2, 10, 20]),
        ("xyxy", [True, 2, 10, 20]),
        ("label", ""),
    ],
)
def test_invalid_detection_is_rejected_with_context(prediction_file, key, value):
    data, save = prediction_file
    data["detections"][0][key] = value
    with pytest.raises(ValueError, match="tracks.json: detection 0"):
        load_prediction_json(save())


@pytest.mark.parametrize("key", ["frame_id", "track_id", "xyxy", "confidence", "label"])
def test_missing_detection_fields(prediction_file, key):
    data, save = prediction_file
    del data["detections"][0][key]
    with pytest.raises(ValueError, match=key):
        load_prediction_json(save())


def test_duplicate_frame_and_identity_rejected(prediction_file):
    data, save = prediction_file
    data["detections"].append({**data["detections"][0], "confidence": 0.1})
    with pytest.raises(ValueError, match="duplicate observation"):
        load_prediction_json(save())


def test_sorting_keeps_different_ids_and_all_confidences(prediction_file):
    data, save = prediction_file
    data["detections"] += [{**data["detections"][0], "track_id": 3, "confidence": 0.0}, {**data["detections"][0], "frame_id": 0, "confidence": 1.0}]
    result = load_prediction_json(save())
    assert [(o.frame_id, o.track_id, o.confidence) for o in result.observations] == [(0, 7, 1.0), (1, 3, 0.0), (1, 7, 0.875)]


def test_out_of_bounds_box_warns_without_clipping(prediction_file):
    data, save = prediction_file
    data["detections"][0]["xyxy"] = [-1, 2, 101, 20]
    result = load_prediction_json(save())
    assert result.observations[0].xyxy == (-1, 2, 101, 20)
    assert "beyond image bounds" in result.warnings[0]


def test_inconsistent_track_label_rejected(prediction_file):
    data, save = prediction_file
    data["detections"].append({**data["detections"][0], "frame_id": 2, "label": "other"})
    with pytest.raises(ValueError, match="inconsistent label"):
        load_prediction_json(save())


@pytest.mark.parametrize("text", ["{", "[]", "null"])
def test_malformed_or_nonobject_json(tmp_path, text):
    path = tmp_path / "invalid.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="invalid.json"):
        load_prediction_json(path)


def test_nonobject_detection(prediction_file):
    data, save = prediction_file
    data["detections"] = [None]
    with pytest.raises(ValueError, match="detection 0"):
        load_prediction_json(save())
