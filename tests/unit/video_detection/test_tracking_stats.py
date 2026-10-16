import pytest

from tennisvision.tasks.video_detection.postprocessing import compute_tracking_stats


def test_statistics_with_missing_frames(detection, tracking_result):
    result = tracking_result(
        [
            detection(1, box=(0, 0, 10, 20), confidence=0.6),
            detection(3, box=(3, 4, 13, 24), confidence=1.0),
            detection(4, track=2),
        ]
    )
    stats = compute_tracking_stats(result)
    assert set(stats) == {1, 2}
    assert stats[1] == pytest.approx(
        {
            "total_conf": 1.6,
            "count": 2,
            "first_frame": 1,
            "last_frame": 3,
            "first_center_x": 5,
            "first_center_y": 10,
            "last_center_x": 8,
            "last_center_y": 14,
            "total_box_area": 400,
            "mean_box_area": 200,
            "total_path_distance": 5,
            "mean_conf": 0.8,
            "duration_frames": 3,
            "presence_ratio": 0.4,
            "duration_ratio": 0.6,
        }
    )
    assert stats[2]["duration_frames"] == 1
    assert stats[2]["total_path_distance"] == 0


@pytest.mark.parametrize("unassigned", [False, True])
def test_no_assigned_tracks(detection, tracking_result, unassigned):
    result = tracking_result([detection(track=None)] if unassigned else [])
    assert compute_tracking_stats(result) == {}


def test_unassigned_detection_extends_observed_timeline(detection, tracking_result):
    stats = compute_tracking_stats(tracking_result([detection(), detection(9, track=None)]))
    assert set(stats) == {1}
    assert stats[1]["presence_ratio"] == pytest.approx(0.1)
