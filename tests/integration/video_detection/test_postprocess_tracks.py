"""Exercise the real CLI/file pipeline; isolate MLflow and video rendering."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tennisvision.tasks.video_detection.scripts import postprocess_tracks


@pytest.fixture
def external_services(monkeypatch):
    mlflow = SimpleNamespace(
        start_run=Mock(side_effect=lambda **kwargs: nullcontext()),
        set_tag=Mock(),
        log_artifact=Mock(),
        log_metrics=Mock(),
    )
    setup = Mock()
    render = Mock(side_effect=lambda **kwargs: kwargs["output_path"])
    monkeypatch.setattr(postprocess_tracks, "mlflow", mlflow)
    monkeypatch.setattr(postprocess_tracks, "setup_mlflow", setup)
    monkeypatch.setattr(postprocess_tracks, "render_tracking_video", render)
    return SimpleNamespace(mlflow=mlflow, setup=setup, render=render)


@pytest.fixture
def tracks_payload():
    def detection(frame, track, confidence=0.8, box=(40, 40, 50, 60)):
        return {
            "frame_id": frame,
            "timestamp_sec": frame / 25,
            "class_id": 0,
            "label": "player",
            "confidence": confidence,
            "xyxy": list(box),
            "track_id": track,
        }

    return {
        "video_path": "missing-synthetic-video.mp4",
        "width": 100,
        "height": 100,
        "fps": 25.0,
        "detections": sorted(
            [
                *[detection(f, 7) for f in range(4)],
                *[detection(f, 8, confidence=0.9) for f in range(2)],
                *[detection(f, 42) for f in range(4, 6)],
                *[detection(f, 99, confidence=0.1, box=(80, 80, 90, 90)) for f in range(6)],
            ],
            key=lambda d: d["frame_id"],
        ),
    }


def run_cli(monkeypatch, source, *arguments):
    monkeypatch.setattr(
        "sys.argv",
        [
            "postprocess_tracks",
            "--tracks",
            str(source),
            "--min-count",
            "2",
            "--min-presence-ratio",
            "0",
            "--min-duplicate-overlap-frames",
            "2",
            "--max-stitch-gap-ratio",
            "1",
            "--max-stitch-overlap",
            "0",
            "--max-tracks",
            "1",
            *arguments,
        ],
    )
    postprocess_tracks.main()


@pytest.mark.parametrize("explicit_output", [False, True], ids=["default-output", "custom-output"])
def test_json_to_postprocessed_artifacts(tmp_path, monkeypatch, external_services, tracks_payload, explicit_output):
    source = tmp_path / "tracks.json"
    original_text = json.dumps(tracks_payload)
    source.write_text(original_text)
    output = tmp_path / "nested" / "output" if explicit_output else tmp_path
    arguments = ["--output-dir", str(output), "--video", "override.mp4"] if explicit_output else []

    run_cli(monkeypatch, source, *arguments)

    processed = json.loads((output / "tracks_postprocessed.json").read_text())
    info = json.loads((output / "postprocessing_info.json").read_text())
    summary = json.loads((output / "summary_postprocessed.json").read_text())
    assert source.read_text() == original_text
    assert {key: value for key, value in processed.items() if key != "detections"} == {
        key: value for key, value in tracks_payload.items() if key != "detections"
    }
    assert [d["frame_id"] for d in processed["detections"]] == list(range(6))
    assert [d["track_id"] for d in processed["detections"]] == [7] * 6
    assert [d["confidence"] for d in processed["detections"]] == [0.9, 0.9, 0.8, 0.8, 0.8, 0.8]
    assert all(d["xyxy"] == [40, 40, 50, 60] for d in processed["detections"])
    # JSON object keys become strings; ID values remain integers.
    assert info["overlap_mapping"] == {"8": 7}
    assert info["stitching_mapping"] == {"42": 7}
    assert info["active_selection"]["selected_track_ids"] == [7]
    assert info["final_stats"]["7"]["count"] == 6
    assert summary == {
        "min_detections_per_frame": 1,
        "max_detections_per_frame": 1,
        "mean_detections_per_frame": 1.0,
        "frames_with_detections": 6,
        "frames_with_detections_ratio": 1.0,
        "unique_track_ids": 1,
    }
    render_args = external_services.render.call_args.kwargs
    assert render_args["video_path"] == Path("override.mp4" if explicit_output else tracks_payload["video_path"])
    assert render_args["output_path"] == output / "video_postprocessed.mp4"
    assert [d.track_id for d in render_args["tracking_result"].detections] == [7] * 6
    external_services.mlflow.log_metrics.assert_called_once_with(summary)
    logged_paths = {call.args[0] for call in external_services.mlflow.log_artifact.call_args_list}
    assert logged_paths == {
        str(output / name)
        for name in (
            "tracks_postprocessed.json",
            "postprocessing_info.json",
            "summary_postprocessed.json",
            "video_postprocessed.mp4",
        )
    }

    # A written result is valid input for a second CLI invocation.
    second_output = tmp_path / "second-pass"
    run_cli(monkeypatch, output / "tracks_postprocessed.json", "--output-dir", str(second_output))
    assert json.loads((second_output / "tracks_postprocessed.json").read_text()) == processed


def test_empty_json_sequence_writes_empty_result(tmp_path, monkeypatch, external_services, tracks_payload):
    tracks_payload["detections"] = []
    source = tmp_path / "empty.json"
    source.write_text(json.dumps(tracks_payload))
    run_cli(monkeypatch, source)
    assert json.loads((tmp_path / "tracks_postprocessed.json").read_text()) == tracks_payload
    summary = json.loads((tmp_path / "summary_postprocessed.json").read_text())
    assert all(value == 0 for value in summary.values())
    info = json.loads((tmp_path / "postprocessing_info.json").read_text())
    assert info["final_stats"] == {}
    assert info["active_selection"]["selected_track_ids"] == []


@pytest.mark.parametrize("contents,error", [("{", json.JSONDecodeError), ('{"detections": []}', KeyError)])
def test_invalid_input_fails_before_external_work(tmp_path, monkeypatch, external_services, contents, error):
    source = tmp_path / "invalid.json"
    source.write_text(contents)
    output = tmp_path / "output"
    with pytest.raises(error):
        run_cli(monkeypatch, source, "--output-dir", str(output))
    assert not output.exists()
    external_services.setup.assert_not_called()
    external_services.mlflow.start_run.assert_not_called()
    external_services.render.assert_not_called()
