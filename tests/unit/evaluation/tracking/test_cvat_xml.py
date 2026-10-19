import xml.etree.ElementTree as ET
from dataclasses import FrozenInstanceError

import pytest

from tennisvision.evaluation.tracking.io import load_cvat_xml
from tennisvision.evaluation.tracking.types import TrackingObservation


@pytest.fixture
def annotation_xml(tmp_path):
    """A complete two-frame sequence; tests edit only the relevant XML fields."""
    root = ET.fromstring("""
    <annotations><version>1.1</version><meta><task>
      <name>synthetic</name><mode>interpolation</mode><size>2</size>
      <start_frame>0</start_frame><stop_frame>1</stop_frame><frame_filter>step=1</frame_filter>
      <original_size><width>100</width><height>80</height></original_size>
    </task></meta><track id="0" label="tennis_player">
      <box frame="0" outside="0" occluded="0" keyframe="1" xtl="10.5" ytl="20" xbr="30" ybr="40"/>
      <box frame="1" outside="0" occluded="1" keyframe="0" xtl="11" ytl="21" xbr="31" ybr="41"/>
    </track></annotations>""")

    def save():
        path = tmp_path / "annotations.xml"
        ET.ElementTree(root).write(path, encoding="utf-8")
        return path

    return root, save


def test_dense_sequence_preserves_ids_coordinates_and_flags(annotation_xml):
    _, save = annotation_xml
    path = save()
    sequence = load_cvat_xml(path)
    assert (sequence.sequence_id, sequence.frame_count, sequence.width, sequence.height) == ("synthetic", 2, 100, 80)
    assert sequence.fps is None
    assert sequence.source_path == path
    assert sequence.source_format == "cvat_video_xml_1.1"
    assert sequence.track_labels == {0: "tennis_player"}
    assert sequence.warnings == ()
    assert sequence.observations == (
        TrackingObservation(0, 0, (10.5, 20, 30, 40), outside=False, occluded=False, keyframe=True),
        TrackingObservation(1, 0, (11, 21, 31, 41), outside=False, occluded=True, keyframe=False),
    )
    # Occlusion and interpolation do not discard a present observation.
    assert len([o for o in sequence.observations if o.outside is False]) == 2
    with pytest.raises(FrozenInstanceError):
        sequence.observations[0].frame_id = 5


@pytest.mark.parametrize("coordinates", [(0, 0, 0, 0), (30, 40, 10, 20), (-10, -10, 200, 200)])
def test_outside_markers_keep_nonvisible_geometry(annotation_xml, coordinates):
    root, save = annotation_xml
    box = root.find("track/box")
    box.set("outside", "1")
    for name, value in zip(("xtl", "ytl", "xbr", "ybr"), coordinates, strict=True):
        box.set(name, str(value))
    result = load_cvat_xml(save())
    assert len(result.observations) == 2
    assert result.observations[0].outside is True
    assert result.observations[0].xyxy == coordinates
    assert result.warnings == ()


def test_missing_records_are_not_filled_or_marked_absent(annotation_xml):
    root, save = annotation_xml
    track = root.find("track")
    track.remove(track.findall("box")[1])
    sequence = load_cvat_xml(save())
    assert len(sequence.observations) == 1
    assert sequence.frame_count == 2
    assert sequence.observations[0].outside is False
    assert "1 frames have no annotation record" in sequence.warnings[0]
    assert "unknown" in sequence.warnings[0]


def test_empty_annotations_are_not_assumed_complete(annotation_xml):
    root, save = annotation_xml
    root.remove(root.find("track"))
    sequence = load_cvat_xml(save())
    assert sequence.observations == ()
    assert sequence.track_labels == {}
    assert sequence.frame_count == 2
    assert "completeness" in sequence.warnings[0]


def test_sorting_does_not_renumber_frames_or_ids(annotation_xml):
    root, save = annotation_xml
    track = root.find("track")
    track.set("id", "9")
    first = track[0]
    track.remove(first)
    track.append(first)
    other = ET.fromstring(ET.tostring(track))
    other.set("id", "3")
    root.append(other)
    sequence = load_cvat_xml(save())
    assert [(o.frame_id, o.track_id) for o in sequence.observations] == [(0, 3), (0, 9), (1, 3), (1, 9)]


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("frame", "-1"),
        ("frame", "0.5"),
        ("frame", "2"),
        ("outside", "true"),
        ("occluded", "2"),
        ("keyframe", "yes"),
        ("xtl", "nan"),
        ("ytl", "inf"),
        ("xbr", "bad"),
        ("xbr", "10.5"),
        ("ybr", "19"),
    ],
)
def test_invalid_box_fields_report_context(annotation_xml, attribute, value):
    root, save = annotation_xml
    root.find("track/box").set(attribute, value)
    with pytest.raises(ValueError, match="track 0"):
        load_cvat_xml(save())


@pytest.mark.parametrize("attribute", ["frame", "outside", "occluded", "keyframe", "xtl", "ytl", "xbr", "ybr"])
def test_missing_box_fields_are_rejected(annotation_xml, attribute):
    root, save = annotation_xml
    del root.find("track/box").attrib[attribute]
    with pytest.raises(ValueError, match="track 0"):
        load_cvat_xml(save())


def test_nonfinite_outside_coordinates_still_rejected(annotation_xml):
    root, save = annotation_xml
    root.find("track/box").attrib.update(outside="1", xtl="nan")
    with pytest.raises(ValueError, match="finite"):
        load_cvat_xml(save())


def test_bounds_and_unexpected_labels_warn_without_changing_data(annotation_xml):
    root, save = annotation_xml
    root.find("track").set("label", "other")
    root.find("track/box").set("xtl", "-5")
    sequence = load_cvat_xml(save())
    assert sequence.observations[0].xyxy[0] == -5
    assert sequence.track_labels == {0: "other"}
    assert any("unexpected label" in warning for warning in sequence.warnings)
    assert any("image bounds" in warning for warning in sequence.warnings)


@pytest.mark.parametrize("kind", ["track", "box"])
def test_duplicate_ids_or_records_are_rejected(annotation_xml, kind):
    root, save = annotation_xml
    parent = root if kind == "track" else root.find("track")
    element = root.find("track") if kind == "track" else root.find("track/box")
    parent.append(ET.fromstring(ET.tostring(element)))
    with pytest.raises(ValueError, match="duplicate"):
        load_cvat_xml(save())


@pytest.mark.parametrize("tag", ["polygon", "points", "skeleton"])
def test_unsupported_shapes_are_not_silently_ignored(annotation_xml, tag):
    root, save = annotation_xml
    ET.SubElement(root.find("track"), tag)
    with pytest.raises(ValueError, match="unsupported shape"):
        load_cvat_xml(save())


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_frame", "1"),
        ("frame_filter", "step=2"),
        ("size", "3"),
        ("size", "0"),
        ("stop_frame", "-1"),
        ("original_size/width", "0"),
        ("original_size/height", "bad"),
        ("mode", "annotation"),
        ("name", ""),
    ],
)
def test_unsupported_or_invalid_task_metadata(annotation_xml, field, value):
    root, save = annotation_xml
    root.find(f"meta/task/{field}").text = value
    with pytest.raises(ValueError):
        load_cvat_xml(save())


@pytest.mark.parametrize("field", ["size", "start_frame", "stop_frame", "original_size", "mode"])
def test_required_metadata_missing(annotation_xml, field):
    root, save = annotation_xml
    task = root.find("meta/task")
    task.remove(task.find(field))
    with pytest.raises(ValueError):
        load_cvat_xml(save())


def test_empty_frame_filter_means_no_subsampling(annotation_xml):
    root, save = annotation_xml
    root.find("meta/task/frame_filter").text = ""
    assert load_cvat_xml(save()).frame_count == 2


@pytest.mark.parametrize("attribute,value", [("id", "-1"), ("id", "abc"), ("label", "")])
def test_invalid_track_metadata(annotation_xml, attribute, value):
    root, save = annotation_xml
    root.find("track").set(attribute, value)
    with pytest.raises(ValueError):
        load_cvat_xml(save())


@pytest.mark.parametrize("contents", ["<broken", "<annotations><version>2.0</version></annotations>", "<other/>"])
def test_invalid_document(tmp_path, contents):
    path = tmp_path / "invalid.xml"
    path.write_text(contents)
    with pytest.raises(ValueError, match="invalid.xml"):
        load_cvat_xml(path)


def test_prediction_metadata_defaults_are_unknown():
    observation = TrackingObservation(0, None, (0, 0, 1, 1), confidence=0.8)
    assert observation.outside is None
    assert observation.occluded is None
    assert observation.keyframe is None
