import json

from tennisvision.evaluation.tracking.io import load_cvat_xml, load_prediction_json
from tennisvision.evaluation.tracking.types import TrackingSequence


def test_xml_and_independent_prediction_files_share_representation(tmp_path):
    xml = tmp_path / "annotations.xml"
    xml.write_text("""<annotations><version>1.1</version><meta><task>
    <name>annotation_task</name><mode>interpolation</mode><size>3</size>
    <start_frame>0</start_frame><stop_frame>2</stop_frame><frame_filter>step=1</frame_filter>
    <original_size><width>100</width><height>80</height></original_size></task></meta>
    <track id="0" label="tennis_player">
    <box frame="0" outside="0" occluded="0" keyframe="1" xtl="10" ytl="20" xbr="30" ybr="40"/>
    <box frame="1" outside="0" occluded="1" keyframe="0" xtl="10" ytl="20" xbr="30" ybr="40"/>
    <box frame="2" outside="1" occluded="0" keyframe="1" xtl="0" ytl="0" xbr="0" ybr="0"/>
    </track></annotations>""")
    truth = load_cvat_xml(xml)
    for name, track_id in [("tracks.json", 7), ("tracks_postprocessed.json", 42)]:
        path = tmp_path / name
        payload = {
            "video_path": "clip.mp4",
            "width": 100,
            "height": 80,
            "fps": 60,
            "processed_frame_count": 3,
            "detections": [{"frame_id": 0, "track_id": track_id, "label": "player", "confidence": 0.95, "xyxy": [10, 20, 30, 40]}],
        }
        original = json.dumps(payload)
        path.write_text(original)
        prediction = load_prediction_json(path)
        assert isinstance(prediction, TrackingSequence)
        assert prediction.frame_count == truth.frame_count == 3
        assert (prediction.width, prediction.height) == (truth.width, truth.height)
        assert prediction.observations[0].xyxy == truth.observations[0].xyxy
        assert prediction.observations[0].frame_id == truth.observations[0].frame_id == 0
        assert prediction.observations[0].track_id == track_id  # No GT-ID remapping.
        assert prediction.observations[0].outside is None
        assert prediction.observations[0].occluded is None
        assert prediction.observations[0].keyframe is None
        assert len(prediction.observations) == 1  # Frames 1 and 2 remain empty.
        assert path.read_text() == original
    assert truth.warnings == ()
