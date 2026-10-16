from dataclasses import replace

import pytest

from tennisvision.tasks.video_detection.postprocessing import TrackPostProcessingConfig, compute_tracking_stats, filter_short_tracks


@pytest.mark.parametrize("count,kept", [(2, False), (3, True), (4, True)])
def test_count_threshold(detection, tracking_result, count, kept):
    result = tracking_result([detection(frame) for frame in range(count)])
    output, info = filter_short_tracks(result, compute_tracking_stats(result), TrackPostProcessingConfig(min_count=3, min_presence_ratio=0))
    assert output == replace(result, detections=result.detections if kept else [])
    assert info["dropped_track_ids"] == ([] if kept else [1])


@pytest.mark.parametrize("ratio,kept", [(0.19, True), (0.2, True), (0.21, False)])
def test_presence_threshold_including_fractional_count(detection, tracking_result, ratio, kept):
    result = tracking_result([detection(0), detection(1), detection(9, track=None)])
    output, info = filter_short_tracks(result, compute_tracking_stats(result), TrackPostProcessingConfig(min_count=1, min_presence_ratio=ratio))
    assert output.detections == (result.detections[:2] if kept else [])
    assert info["total_frames"] == 10


def test_filters_only_short_track(detection, tracking_result):
    result = tracking_result([detection(0, track=2), detection(0), detection(1)])
    output, info = filter_short_tracks(result, compute_tracking_stats(result), TrackPostProcessingConfig(min_count=2, min_presence_ratio=0))
    assert [d.track_id for d in output.detections] == [1, 1]
    assert info["dropped_track_ids"] == [2]


def test_empty_filter(tracking_result):
    result = tracking_result()
    output, info = filter_short_tracks(result, {}, TrackPostProcessingConfig())
    assert output == result
    assert info == {}
