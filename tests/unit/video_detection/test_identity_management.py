from dataclasses import replace

import pytest

from tennisvision.tasks.video_detection.postprocessing import (
    TrackPostProcessingConfig,
    apply_track_stitching,
    box_iou,
    compute_tracking_stats,
    deduplicate_track_frame_detections,
    get_near_frame_edge,
    group_detections_by_track_and_frame,
    is_on_same_frame_side,
    merge_overlapping_duplicate_tracks,
    stitch_tracks,
)


@pytest.mark.parametrize(
    "other,expected",
    [
        ((0, 0, 10, 10), 1),
        ((5, 0, 15, 10), 1 / 3),
        ((10, 0, 20, 10), 0),
        ((20, 20, 30, 30), 0),
        ((0, 0, 0, 0), 0),
        ((10, 10, 0, 0), 0),
    ],
)
def test_iou(other, expected):
    box = (0, 0, 10, 10)
    assert box_iou(box, other) == pytest.approx(expected)
    assert box_iou(other, box) == pytest.approx(expected)


def test_two_zero_area_boxes():
    assert box_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0


@pytest.mark.parametrize("x,y,edge", [(8, 50, "left"), (92, 50, "right"), (50, 8, "top"), (50, 92, "bottom"), (50, 50, None)])
def test_frame_edges(x, y, edge):
    assert get_near_frame_edge(x, y, 100, 100) == edge


@pytest.mark.parametrize(
    "edge,x,y,expected",
    [
        ("left", 40, 50, True),
        ("left", 41, 50, False),
        ("right", 60, 50, True),
        ("right", 59, 50, False),
        ("top", 50, 40, True),
        ("top", 50, 41, False),
        ("bottom", 50, 60, True),
        ("bottom", 50, 59, False),
        ("unknown", 50, 50, False),
    ],
)
def test_frame_sides(edge, x, y, expected):
    assert is_on_same_frame_side(edge, x, y, 100, 100, 0.4) is expected


@pytest.fixture
def stitch_config():
    # Explicit limits isolate each boundary from video-length scaling.
    return TrackPostProcessingConfig(
        max_stitch_frame_gap=3,
        max_stitch_gap_ratio=1,
        max_stitch_overlap=0,
        max_stitch_center_distance=10,
        max_stitch_center_distance_ratio=1,
        max_reentry_frame_gap=10,
        max_reentry_gap_ratio=1,
        max_reentry_center_distance=30,
        max_reentry_center_distance_ratio=1,
    )


@pytest.mark.parametrize(
    "gap,shift,area_scale,merged",
    [
        (2, 0, 1, True),
        (3, 0, 1, True),
        (4, 0, 1, False),
        (2, 10, 1, True),
        (2, 11, 1, False),
        (2, 0, 2, True),
        (2, 0, 2.1, False),
        (2, 0, 0, False),
    ],
)
def test_stitching_limits(detection, tracking_result, stitch_config, gap, shift, area_scale, merged):
    # Change width around a fixed center to isolate area from distance.
    result = tracking_result(
        [
            detection(0),
            detection(1),
            detection(1 + gap, track=2, box=(45 + shift - 5 * area_scale, 40, 45 + shift + 5 * area_scale, 60)),
        ]
    )
    mapping = stitch_tracks(result, compute_tracking_stats(result), stitch_config)
    assert mapping == ({2: 1} if merged else {})


@pytest.mark.parametrize("new_x,merged", [(5, True), (30, True), (85, False)])
def test_reentry_uses_longer_gap_on_same_side(detection, tracking_result, stitch_config, new_x, merged):
    result = tracking_result(
        [
            detection(0, box=(0, 40, 10, 60)),
            detection(8, track=2, box=(new_x - 5, 40, new_x + 5, 60)),
        ]
    )
    assert stitch_tracks(result, compute_tracking_stats(result), stitch_config) == ({2: 1} if merged else {})


def test_closest_predecessor_wins(detection, tracking_result, stitch_config):
    result = tracking_result([detection(0), detection(0, track=2, box=(49, 40, 59, 60)), detection(2, track=3, box=(48, 40, 58, 60))])
    # Predecessors are beyond each other's distance threshold.
    cfg = replace(stitch_config, max_stitch_center_distance=2)
    assert stitch_tracks(result, compute_tracking_stats(result), cfg) == {3: 2}


def test_chained_mapping_preserves_fields_and_input(detection, tracking_result):
    result = tracking_result([detection(track=3), detection(1, track=2), detection(2, track=None), detection(3, track=99)])
    original = list(result.detections)
    output = apply_track_stitching(result, {3: 2, 2: 1})
    assert output == replace(result, detections=[replace(original[0], track_id=1), replace(original[1], track_id=1), *original[2:]])
    assert result.detections == original


def test_deduplication_and_grouping(detection, tracking_result):
    low = detection(1, confidence=0.2)
    high = detection(1, confidence=0.9)
    other = detection(1, track=2)
    first = detection(0)
    result = tracking_result([low, other, high, first, detection(track=None)])
    assert group_detections_by_track_and_frame(result) == {1: {1: [low, high], 0: [first]}, 2: {1: [other]}}
    assert deduplicate_track_frame_detections(result) == replace(result, detections=[first, high, other])


@pytest.mark.parametrize("overlap,shift,merged", [(1, 0, False), (2, 0, True), (3, 0, True), (2, 5, True), (2, 6, False), (2, 30, False)])
def test_duplicate_overlap_and_iou_limits(detection, tracking_result, overlap, shift, merged):
    result = tracking_result(
        [detection(frame) for frame in range(4)] + [detection(frame, track=2, box=(40 + shift, 40, 50 + shift, 60)) for frame in range(overlap)]
    )
    cfg = TrackPostProcessingConfig(min_duplicate_overlap_frames=2, min_duplicate_iou=1 / 3)
    assert merge_overlapping_duplicate_tracks(result, compute_tracking_stats(result), cfg) == ({2: 1} if merged else {})


def test_duplicate_matching_uses_highest_confidence_box(detection, tracking_result):
    result = tracking_result([detection(), detection(track=2), detection(track=2, box=(80, 80, 90, 90), confidence=0.1)])
    cfg = TrackPostProcessingConfig(min_duplicate_overlap_frames=1)
    assert merge_overlapping_duplicate_tracks(result, compute_tracking_stats(result), cfg) == {1: 2}


def test_empty_identity_operations(tracking_result, stitch_config):
    result = tracking_result()
    assert stitch_tracks(result, {}, stitch_config) == {}
    assert merge_overlapping_duplicate_tracks(result, {}, stitch_config) == {}
    assert apply_track_stitching(result, {}) == result
    assert deduplicate_track_frame_detections(result) == result


@pytest.mark.parametrize("gap_ratio,merged", [(0.1, False), (2 / 3, True)])
def test_relative_gap_limit_can_be_stricter_than_absolute_limit(detection, tracking_result, stitch_config, gap_ratio, merged):
    result = tracking_result([detection(0), detection(2, track=2)])
    cfg = replace(stitch_config, max_stitch_gap_ratio=gap_ratio)
    assert stitch_tracks(result, compute_tracking_stats(result), cfg) == ({2: 1} if merged else {})


@pytest.mark.parametrize("distance_ratio,merged", [(0.01, False), (0.1, True)])
def test_relative_distance_limit(detection, tracking_result, stitch_config, distance_ratio, merged):
    result = tracking_result([detection(0), detection(2, track=2, box=(43, 40, 53, 60))])
    cfg = replace(stitch_config, max_stitch_center_distance_ratio=distance_ratio)
    assert stitch_tracks(result, compute_tracking_stats(result), cfg) == ({2: 1} if merged else {})


@pytest.mark.parametrize("overlap,merged", [(0, False), (1, True)])
def test_small_temporal_overlap(detection, tracking_result, stitch_config, overlap, merged):
    result = tracking_result(
        [
            *[detection(f) for f in range(4)],
            *[detection(f, track=2) for f in range(2, 6)],
        ]
    )
    cfg = replace(stitch_config, max_stitch_overlap=overlap, max_stitch_overlap_ratio=1)
    assert stitch_tracks(result, compute_tracking_stats(result), cfg) == ({2: 1} if merged else {})
