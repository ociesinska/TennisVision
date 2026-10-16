from dataclasses import replace

import pytest

from tennisvision.tasks.video_detection.postprocessing import (
    TrackPostProcessingConfig,
    compute_active_track_scores,
    compute_tracking_stats,
    select_active_tracks,
)


def test_score_uses_presence_confidence_motion_and_area(tracking_result):
    # A 30x40 image has diagonal 50 and area 1200:
    # 2 * .5 + .8 + 3 * (10 / 50) + 120 / 1200 = 2.5.
    result = tracking_result(width=30, height=40)
    stats = {7: {"presence_ratio": 0.5, "mean_conf": 0.8, "total_path_distance": 10, "mean_box_area": 120}}
    assert compute_active_track_scores(result, stats) == pytest.approx({7: 2.5})


@pytest.mark.parametrize("limit,selected", [(None, [7, 42, 99]), (0, []), (1, [7]), (2, [7, 42]), (5, [7, 42, 99])])
def test_selection_keeps_original_identities(detection, tracking_result, limit, selected):
    result = tracking_result(
        [
            detection(0, track=99, confidence=0.1),
            detection(0, track=42, confidence=0.5),
            detection(0, track=7, confidence=0.9),
        ]
    )
    output, info = select_active_tracks(result, compute_tracking_stats(result), TrackPostProcessingConfig(max_tracks=limit))
    assert output == replace(result, detections=[d for d in result.detections if d.track_id in selected])
    assert info["selected_track_ids"] == (None if limit is None else selected)
    assert info["dropped_track_ids"] == sorted({7, 42, 99} - set(selected))


def test_empty_scores_and_selection(tracking_result):
    result = tracking_result()
    assert compute_active_track_scores(result, {}) == {}
    output, info = select_active_tracks(result, {}, TrackPostProcessingConfig(max_tracks=2))
    assert output == result
    assert info["selected_track_ids"] == []
