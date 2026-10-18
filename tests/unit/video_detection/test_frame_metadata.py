from dataclasses import replace

import pytest

from tennisvision.tasks.video_detection.api.schemas import VideoTrackingResponse
from tennisvision.tasks.video_detection.postprocessing import (
    TrackPostProcessingConfig,
    compute_tracking_stats,
    filter_short_tracks,
    postprocess_tracking_result,
    stitch_tracks,
)
from tennisvision.tasks.video_detection.tracking import compute_tracking_summary
from tennisvision.tasks.video_detection.types import VideoTrackingResult


def test_trailing_empty_frames_are_included(detection, tracking_result):
    result = replace(tracking_result([detection(0), detection(1)]), processed_frame_count=10, source_frame_count=100)
    summary = compute_tracking_summary(result)
    assert summary == {
        "min_detections_per_frame": 0,
        "max_detections_per_frame": 1,
        "mean_detections_per_frame": 0.2,
        "frames_with_detections": 2,
        "frames_with_detections_ratio": 0.2,
        "unique_track_ids": 1,
    }
    stats = compute_tracking_stats(result)
    assert stats[1]["presence_ratio"] == 0.2
    assert stats[1]["duration_ratio"] == 0.2
    output, _ = filter_short_tracks(result, stats, TrackPostProcessingConfig(min_count=1, min_presence_ratio=0.3))
    assert output.detections == []
    assert output.processed_frame_count == 10


@pytest.mark.parametrize("count", [0, 10])
def test_no_detections_with_known_processed_count(tracking_result, count):
    result = replace(tracking_result(), processed_frame_count=count)
    assert all(value == 0 for value in compute_tracking_summary(result).values())
    assert compute_tracking_stats(result) == {}


def test_duplicates_do_not_inflate_presence(detection, tracking_result):
    result = replace(tracking_result([detection(), detection()]), processed_frame_count=2)
    stats = compute_tracking_stats(result)[1]
    assert stats["count"] == 2
    assert stats["presence_ratio"] == 0.5
    assert compute_tracking_summary(result)["frames_with_detections_ratio"] == 0.5


def test_unsorted_input_keeps_chronological_duration(detection, tracking_result):
    result = replace(tracking_result([detection(3), detection(1)]), processed_frame_count=10)
    stats = compute_tracking_stats(result)[1]
    assert stats["duration_frames"] == 3
    assert stats["duration_ratio"] == 0.3
    assert [d.frame_id for d in result.detections] == [3, 1]


def test_stitch_limits_use_full_processed_timeline(detection, tracking_result):
    result = replace(tracking_result([detection(0), detection(2, track=2)]), processed_frame_count=100)
    cfg = TrackPostProcessingConfig(max_stitch_overlap=0, max_stitch_gap_ratio=0.03)
    assert stitch_tracks(result, compute_tracking_stats(result), cfg) == {2: 1}


def test_legacy_fallback_survives_removal_of_last_detection(detection, tracking_result):
    result = tracking_result([detection(0), detection(1), detection(9, track=2)])
    output, info = postprocess_tracking_result(result, TrackPostProcessingConfig(min_count=2, min_presence_ratio=0))
    assert output.processed_frame_count is None
    assert output.inferred_frame_count == 10
    assert output.statistics_frame_count == 10
    assert info["frame_metadata"]["basis"] == "detection_span_fallback"
    assert info["final_stats"][1]["presence_ratio"] == 0.2
    assert compute_tracking_summary(output)["frames_with_detections_ratio"] == 0.2


@pytest.mark.parametrize("field", ["processed_frame_count", "source_frame_count", "inferred_frame_count"])
@pytest.mark.parametrize("value", [-1, 2.5, True, "10"])
def test_invalid_counts_are_rejected(tracking_result, field, value):
    with pytest.raises(ValueError, match=field):
        replace(tracking_result(), **{field: value})


@pytest.mark.parametrize("count", [0, 2])
def test_detection_outside_processed_frames_is_rejected(detection, tracking_result, count):
    with pytest.raises(ValueError, match="processed_frame_count"):
        replace(tracking_result([detection(2)]), processed_frame_count=count)


@pytest.mark.parametrize("frame", [-1, 0.5, True])
def test_invalid_frame_ids_are_rejected(detection, tracking_result, frame):
    with pytest.raises(ValueError, match="frame_id"):
        tracking_result([detection(frame)])


def test_source_count_never_substitutes_for_processed_count(detection):
    result = VideoTrackingResult("legacy.mp4", 100, 100, 25, [detection(1)], source_frame_count=100)
    assert result.processed_frame_count is None
    assert result.statistics_frame_count == 2
    assert result.frame_count_basis == "detection_span_fallback"


@pytest.mark.parametrize("processed", [None, 0, 100])
def test_api_exposes_counts_and_provenance(tracking_result, processed):
    result = replace(tracking_result(), processed_frame_count=processed, source_frame_count=120)
    response = VideoTrackingResponse.from_result(result, "/video.mp4", "example").model_dump()
    assert response["processed_frame_count"] == processed
    assert response["source_frame_count"] == 120
    assert response["frame_count_basis"] == result.frame_count_basis
