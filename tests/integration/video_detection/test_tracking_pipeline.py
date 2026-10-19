import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from tennisvision.tasks.video_detection.scripts import track_and_postprocess_video as pipeline
from tennisvision.tasks.video_detection.types import VideoTrackDetection, VideoTrackingResult


def test_pipeline_saves_raw_diagnostics_before_filtering(tmp_path, monkeypatch):
    raw = VideoTrackingResult(
        video_path="synthetic.mp4",
        width=100,
        height=100,
        fps=25,
        detections=[VideoTrackDetection(0, 0, 0, "player", 0.8, (40, 40, 50, 60), 7)],
        processed_frame_count=10,
        source_frame_count=10,
    )
    monkeypatch.setattr(pipeline, "track_video", Mock(return_value=raw))
    monkeypatch.setattr(pipeline, "setup_mlflow", Mock())
    monkeypatch.setattr(
        pipeline,
        "mlflow",
        SimpleNamespace(
            start_run=Mock(return_value=nullcontext()),
            set_tag=Mock(),
            log_dict=Mock(),
            log_artifact=Mock(),
            log_metrics=Mock(),
        ),
    )
    monkeypatch.setattr(pipeline, "render_tracking_video", Mock(side_effect=lambda **kw: kw["output_path"]))
    monkeypatch.setattr(
        "sys.argv",
        [
            "track_and_postprocess_video",
            "--video",
            "synthetic.mp4",
            "--output-dir",
            str(tmp_path),
            "--min-count",
            "2",
            "--no-save-video",
        ],
    )
    actual_postprocess = pipeline.postprocess_tracking_result

    def check_raw_files_then_postprocess(result, cfg):
        output = next(tmp_path.iterdir())
        assert json.loads((output / "summary_raw.json").read_text())["unique_track_ids"] == 1
        assert json.loads((output / "stats_raw.json").read_text())["7"]["count"] == 1
        return actual_postprocess(result, cfg)

    monkeypatch.setattr(pipeline, "postprocess_tracking_result", check_raw_files_then_postprocess)
    pipeline.main()

    output = next(tmp_path.iterdir())
    assert len(json.loads((output / "tracks.json").read_text())["detections"]) == 1
    assert json.loads((output / "tracks_postprocessed.json").read_text())["detections"] == []
    assert json.loads((output / "summary_raw.json").read_text())["frames_with_detections_ratio"] == 0.1
    assert json.loads((output / "summary_postprocessed.json").read_text())["frames_with_detections_ratio"] == 0
