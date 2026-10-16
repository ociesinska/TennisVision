from copy import deepcopy
from dataclasses import replace

import pytest

from tennisvision.tasks.video_detection.postprocessing import TrackPostProcessingConfig, postprocess_tracking_result
from tennisvision.tasks.video_detection.types import VideoTrackDetection, VideoTrackingResult


@pytest.mark.parametrize("kind", ["empty", "unassigned", "too_short"])
def test_pipeline_without_surviving_tracks(detection, tracking_result, kind):
    detections = {"empty": [], "unassigned": [detection(track=None)], "too_short": [detection()]}[kind]
    result = tracking_result(detections)
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
