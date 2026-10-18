from copy import deepcopy
from dataclasses import replace

import pytest

from tennisvision.tasks.video_detection.postprocessing import (
    TrackPostProcessingConfig,
    apply_track_stitching,
    compute_tracking_stats,
    deduplicate_track_frame_detections,
    filter_short_tracks,
    postprocess_tracking_result,
    select_active_tracks,
)
from tennisvision.tasks.video_detection.types import VideoTrackDetection, VideoTrackingResult


@pytest.mark.parametrize("kind", ["empty", "unassigned", "too_short"])
def test_pipeline_without_surviving_tracks(detection, tracking_result, kind):
    detections = {"empty": [], "unassigned": [detection(track=None)], "too_short": [detection()]}[kind]
    result = replace(tracking_result(detections), processed_frame_count=100, source_frame_count=120)
    output, info = postprocess_tracking_result(result, TrackPostProcessingConfig(max_tracks=2))
    assert isinstance(output, VideoTrackingResult)
    assert output == replace(result, detections=[])
    assert info["final_stats"] == {}
    assert info["active_selection"]["selected_track_ids"] == []


def test_pipeline_merges_stitches_deduplicates_then_selects(detection, tracking_result):
    # Track 2 duplicates 1; track 3 continues it. Track 9 is irrelevant.
    # Merging before stitching avoids choosing the duplicate as predecessor.
    # Selection before stitching would lose the shorter continuation.
    result = tracking_result(
        sorted(
            [
                *[detection(f, track=1) for f in range(4)],
                *[detection(f, track=2, confidence=0.9) for f in range(2)],
                *[detection(f, track=3) for f in range(4, 6)],
                *[detection(f, track=9, box=(80, 80, 90, 90), confidence=0.1) for f in range(6)],
                detection(2, track=99),
            ],
            key=lambda d: d.frame_id,
        )
    )
    result = replace(result, processed_frame_count=100, source_frame_count=120)
    original = deepcopy(result)
    cfg = TrackPostProcessingConfig(
        min_count=2,
        min_presence_ratio=0,
        min_duplicate_overlap_frames=2,
        max_stitch_gap_ratio=1,
        max_stitch_overlap=0,
        max_tracks=1,
    )
    output, info = postprocess_tracking_result(result, cfg)
    assert result == original
    expected = [detection(f, confidence=0.9 if f < 2 else 0.8) for f in range(6)]
    assert output == replace(result, detections=expected)
    assert all(isinstance(d, VideoTrackDetection) for d in output.detections)
    assert info["filtering"]["dropped_track_ids"] == [99]
    assert info["overlap_mapping"] == {2: 1}
    assert info["stitching_mapping"] == {3: 1}
    assert info["active_selection"]["selected_track_ids"] == [1]
    assert info["active_selection"]["dropped_track_ids"] == [9]
    assert set(info["final_stats"]) == {1}
    assert info["final_stats"][1]["count"] == 6


def test_short_fragments_are_filtered_before_stitching(detection, tracking_result):
    result = tracking_result([detection(0), detection(1, track=2)])
    output, info = postprocess_tracking_result(result, TrackPostProcessingConfig(min_count=2, max_stitch_gap_ratio=1))
    assert output.detections == []
    assert info["filtering"]["dropped_track_ids"] == [1, 2]
    assert info["stitching_mapping"] == {}


@pytest.mark.parametrize("counts", [(100, 120), (100, None), (None, None)], ids=["known", "unknown-source", "legacy"])
@pytest.mark.parametrize("operation", ["filter", "stitch", "deduplicate", "select"])
def test_each_cleanup_preserves_metadata_and_input(detection, tracking_result, counts, operation):
    low = detection(0, track=2, confidence=0.2)
    high = detection(0, track=2, confidence=0.9)
    other = detection(1, track=3)
    result = replace(
        tracking_result([low, high, other]),
        processed_frame_count=counts[0],
        source_frame_count=counts[1],
    )
    original = deepcopy(result)
    stats = compute_tracking_stats(result)

    if operation == "filter":
        output, _ = filter_short_tracks(result, stats, TrackPostProcessingConfig(min_count=2, min_presence_ratio=0))
        expected = [low, high]
    elif operation == "stitch":
        output = apply_track_stitching(result, {3: 2})
        expected = [low, high, replace(other, track_id=2)]
    elif operation == "deduplicate":
        output = deduplicate_track_frame_detections(result)
        expected = [high, other]
    else:
        output, _ = select_active_tracks(result, stats, TrackPostProcessingConfig(max_tracks=0))
        expected = []

    assert output == replace(original, detections=expected)
    assert result == original
    assert output is not result


def test_pipeline_preserves_explicit_zero_processed_frames(tracking_result):
    result = replace(tracking_result(), processed_frame_count=0, source_frame_count=120)
    output, _ = postprocess_tracking_result(result, TrackPostProcessingConfig())
    assert output == result
