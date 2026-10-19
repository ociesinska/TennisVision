"""Load dense, zero-based CVAT video XML without inferring absent players."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from tennisvision.evaluation.tracking.types import TrackingObservation, TrackingSequence


def _integer(value: str | None, context: str, minimum: int = 0) -> int:
    try:
        number = int(value) if value is not None else None
    except ValueError:
        number = None
    if number is None or number < minimum:
        raise ValueError(f"{context}: expected an integer >= {minimum}, got {value!r}")
    return number


def _flag(box: ET.Element, name: str, context: str) -> bool:
    value = box.get(name)
    if value not in ("0", "1"):
        raise ValueError(f"{context}: {name} must be '0' or '1', got {value!r}")
    return value == "1"


def load_cvat_xml(path: str | Path) -> TrackingSequence:
    """Load CVAT 1.1 interpolation annotations.

    All records are retained, including outside markers. Consumers must exclude
    outside=True from matching; neither occlusion nor keyframe status excludes a
    present observation. Warnings must be reviewed before treating unrecorded
    frames as eligible benchmark frames. No interpolation or clipping is applied.
    """
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{path}: malformed CVAT XML: {exc}") from exc
    if root.tag != "annotations" or root.findtext("version") != "1.1":
        raise ValueError(f"{path}: expected CVAT annotations version 1.1")
    task = root.find("meta/task")
    if task is None or task.findtext("mode") != "interpolation":
        raise ValueError(f"{path}: expected video task metadata with mode=interpolation")
    sequence_id = task.findtext("name")
    if not sequence_id or not sequence_id.strip():
        raise ValueError(f"{path}: missing task name")
    start = _integer(task.findtext("start_frame"), f"{path}: start_frame")
    stop = _integer(task.findtext("stop_frame"), f"{path}: stop_frame")
    frame_count = _integer(task.findtext("size"), f"{path}: size", minimum=1)
    frame_filter = (task.findtext("frame_filter") or "").strip()
    if start != 0 or frame_filter not in ("", "step=1"):
        raise ValueError(f"{path}: only start_frame=0 and frame step=1 are supported")
    if stop + 1 != frame_count:
        raise ValueError(f"{path}: task size {frame_count} disagrees with frame range 0–{stop}")
    width = _integer(task.findtext("original_size/width"), f"{path}: width", minimum=1)
    height = _integer(task.findtext("original_size/height"), f"{path}: height", minimum=1)

    labels: dict[int, str] = {}
    observations = []
    warnings = []
    for child in root:
        if child.tag not in {"version", "meta", "track"}:
            raise ValueError(f"{path}: unsupported annotation element <{child.tag}>")
    for track in root.findall("track"):
        track_id = _integer(track.get("id"), f"{path}: track id")
        context = f"{path}: track {track_id}"
        if track_id in labels:
            raise ValueError(f"{context}: duplicate track ID")
        label = track.get("label")
        if not label or not label.strip():
            raise ValueError(f"{context}: missing label")
        labels[track_id] = label
        if label != "tennis_player":
            warnings.append(f"Track {track_id}: unexpected label {label!r}; retained for explicit review")
        frames = set()
        outside_bounds = 0
        for box in track:
            if box.tag != "box":
                raise ValueError(f"{context}: unsupported shape <{box.tag}>")
            frame_id = _integer(box.get("frame"), f"{context}: frame")
            box_context = f"{context}, frame {frame_id}"
            if frame_id >= frame_count:
                raise ValueError(f"{box_context}: frame outside declared range 0–{stop}")
            if frame_id in frames:
                raise ValueError(f"{box_context}: duplicate observation")
            frames.add(frame_id)
            outside = _flag(box, "outside", box_context)
            occluded = _flag(box, "occluded", box_context)
            keyframe = _flag(box, "keyframe", box_context)
            try:
                xyxy = tuple(float(box.attrib[name]) for name in ("xtl", "ytl", "xbr", "ybr"))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{box_context}: missing or non-numeric box coordinates") from exc
            if not all(math.isfinite(value) for value in xyxy):
                raise ValueError(f"{box_context}: box coordinates must be finite")
            x1, y1, x2, y2 = xyxy
            # Outside markers need not describe a valid visible rectangle.
            if not outside:
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"{box_context}: present box must have positive width and height")
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    outside_bounds += 1
            observations.append(
                TrackingObservation(
                    frame_id=frame_id,
                    track_id=track_id,
                    xyxy=xyxy,
                    outside=outside,
                    occluded=occluded,
                    keyframe=keyframe,
                )
            )
        if outside_bounds:
            warnings.append(f"Track {track_id}: {outside_bounds} present boxes extend beyond image bounds; coordinates retained")
        if len(frames) < frame_count:
            warnings.append(
                f"Track {track_id}: {frame_count - len(frames)} frames have no annotation record; absence and evaluation eligibility are unknown"
            )
    if not labels:
        warnings.append("No tracks annotated; annotation completeness must be confirmed before evaluation")
    observations.sort(key=lambda observation: (observation.frame_id, observation.track_id))
    return TrackingSequence(
        sequence_id=sequence_id,
        frame_count=frame_count,
        width=width,
        height=height,
        fps=None,
        track_labels=labels,
        observations=tuple(observations),
        source_path=path,
        source_format="cvat_video_xml_1.1",
        warnings=tuple(warnings),
    )
