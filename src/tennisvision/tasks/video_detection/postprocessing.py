from dataclasses import dataclass

from tennisvision.tasks.video_detection.types import VideoTrackDetection, VideoTrackingResult


@dataclass
class TrackPostProcessingConfig:
    min_count: int = 10
    min_presence_ratio: float = 0.02
    min_mean_conf: float = 0.5
    min_mean_box_area_ratio: float = 0.0002
    min_total_path_distance_ratio: float = 0.005
    max_stitch_frame_gap: int = 30
    max_stitch_gap_ratio: float = 0.03
    max_stitch_overlap: int = 10
    max_stitch_overlap_ratio: float = 0.01
    max_stitch_center_distance: float = 80.0
    max_stitch_center_distance_ratio: float = 0.05
    max_stitch_area_ratio: float = 2.0
    min_duplicate_overlap_frames: int = 5
    min_duplicate_iou: float = 0.5
    max_tracks: int | None = None
    edge_margin_ratio: float = 0.08
    max_reentry_frame_gap: int = 180
    max_reentry_gap_ratio: float = 0.25
    max_reentry_center_distance: float = 700.0
    max_reentry_center_distance_ratio: float = 0.35
    max_reentry_area_ratio: float = 3.0
    reentry_side_ratio: float = 0.4


def compute_tracking_stats(result: VideoTrackingResult) -> dict[int, dict[str, int | float]]:
    """Compute per-track summary statistics used by later cleanup steps."""

    tracking_stats: dict[int, dict[str, int | float]] = {}

    if not result.detections:
        return {}

    for detection in result.detections:
        if detection.track_id is None:
            continue

        curr_track_id = detection.track_id

        x1, y1, x2, y2 = detection.xyxy
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        box_area = (x2 - x1) * (y2 - y1)

        if curr_track_id not in tracking_stats:
            tracking_stats[curr_track_id] = {
                "total_conf": 0.0,
                "count": 0,
                "first_frame": detection.frame_id,
                "first_center_x": center_x,
                "first_center_y": center_y,
                "last_frame": detection.frame_id,
                "last_center_x": center_x,
                "last_center_y": center_y,
                "total_box_area": 0.0,
                "total_path_distance": 0.0,
            }

        tracking_stats[curr_track_id]["total_conf"] += detection.confidence
        tracking_stats[curr_track_id]["count"] += 1
        tracking_stats[curr_track_id]["last_frame"] = detection.frame_id
        tracking_stats[curr_track_id]["total_box_area"] += box_area

        prev_center_x = tracking_stats[curr_track_id]["last_center_x"]
        prev_center_y = tracking_stats[curr_track_id]["last_center_y"]

        dx = center_x - prev_center_x
        dy = center_y - prev_center_y

        tracking_stats[curr_track_id]["total_path_distance"] += (dx**2 + dy**2) ** 0.5

        tracking_stats[curr_track_id]["last_center_x"] = center_x
        tracking_stats[curr_track_id]["last_center_y"] = center_y

    if not tracking_stats:
        return {}

    total_frames = max(detection.frame_id for detection in result.detections) + 1

    for track_id in tracking_stats:
        stats = tracking_stats[track_id]
        stats["mean_conf"] = stats["total_conf"] / stats["count"]
        stats["mean_box_area"] = stats["total_box_area"] / stats["count"]
        stats["duration_frames"] = stats["last_frame"] - stats["first_frame"] + 1
        stats["presence_ratio"] = stats["count"] / total_frames
        stats["duration_ratio"] = stats["duration_frames"] / total_frames

    return tracking_stats


def filter_short_tracks(
    result: VideoTrackingResult, stats: dict[int, dict[str, int | float]], cfg: TrackPostProcessingConfig
) -> tuple[VideoTrackingResult, dict]:
    """Remove very short tracks that are likely unstable false positives."""

    if not result.detections:
        return result, {}

    total_frames = max(d.frame_id for d in result.detections) + 1
    min_required_count = max(cfg.min_count, int(total_frames * cfg.min_presence_ratio))

    drop_track_ids = set()

    for track_id, track_stats in stats.items():
        if track_stats["count"] < min_required_count:
            drop_track_ids.add(track_id)

        if track_stats["presence_ratio"] < cfg.min_presence_ratio:
            drop_track_ids.add(track_id)

    filtered_detections = [
        detection for detection in result.detections if detection.track_id is not None and detection.track_id not in drop_track_ids
    ]

    filtering_info = {
        "total_frames": total_frames,
        "min_required_count": min_required_count,
        "dropped_track_ids": sorted(drop_track_ids),
    }

    new_results = VideoTrackingResult(
        video_path=result.video_path, width=result.width, height=result.height, fps=result.fps, detections=filtered_detections
    )

    return new_results, filtering_info


def get_near_frame_edge(
    center_x: float,
    center_y: float,
    width: int,
    height: int,
    margin_ratio: float = 0.08,
) -> str | None:
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio

    if center_x <= margin_x:
        return "left"
    if center_x >= width - margin_x:
        return "right"
    if center_y <= margin_y:
        return "top"
    if center_y >= height - margin_y:
        return "bottom"

    return None


def is_on_same_frame_side(
    edge: str,
    center_x: float,
    center_y: float,
    width: int,
    height: int,
    side_ratio: float,
) -> bool:
    """Check whether a point is on the same broad image side as an edge."""

    if edge == "left":
        return center_x <= width * side_ratio
    if edge == "right":
        return center_x >= width * (1 - side_ratio)
    if edge == "top":
        return center_y <= height * side_ratio
    if edge == "bottom":
        return center_y >= height * (1 - side_ratio)

    return False


def stitch_tracks(result: VideoTrackingResult, stats: dict[int, dict[str, int | float]], cfg: TrackPostProcessingConfig) -> dict[int, int]:
    """Find adjacent track fragments that likely belong to the same object."""

    if not result.detections or not stats:
        return {}

    best_candidates: dict[int, tuple[int, float]] = {}

    total_frames = max(d.frame_id for d in result.detections) + 1
    image_diag = (result.width**2 + result.height**2) ** 0.5

    max_overlap = min(
        cfg.max_stitch_overlap,
        int(total_frames * cfg.max_stitch_overlap_ratio),
    )

    normal_max_gap = min(
        cfg.max_stitch_frame_gap,
        int(total_frames * cfg.max_stitch_gap_ratio),
    )
    normal_max_center_distance = min(
        cfg.max_stitch_center_distance,
        image_diag * cfg.max_stitch_center_distance_ratio,
    )

    reentry_max_gap = min(
        cfg.max_reentry_frame_gap,
        int(total_frames * cfg.max_reentry_gap_ratio),
    )
    reentry_max_center_distance = min(
        cfg.max_reentry_center_distance,
        image_diag * cfg.max_reentry_center_distance_ratio,
    )

    for old_track_id, old_stats in stats.items():
        for new_track_id, new_stats in stats.items():
            if old_track_id == new_track_id:
                continue

            frame_gap = new_stats["first_frame"] - old_stats["last_frame"]

            old_edge = get_near_frame_edge(
                center_x=old_stats["last_center_x"],
                center_y=old_stats["last_center_y"],
                width=result.width,
                height=result.height,
                margin_ratio=cfg.edge_margin_ratio,
            )

            new_edge = get_near_frame_edge(
                center_x=new_stats["first_center_x"],
                center_y=new_stats["first_center_y"],
                width=result.width,
                height=result.height,
                margin_ratio=cfg.edge_margin_ratio,
            )

            is_reentry_candidate = old_edge is not None and (
                old_edge == new_edge
                or is_on_same_frame_side(
                    edge=old_edge,
                    center_x=new_stats["first_center_x"],
                    center_y=new_stats["first_center_y"],
                    width=result.width,
                    height=result.height,
                    side_ratio=cfg.reentry_side_ratio,
                )
            )

            if is_reentry_candidate:
                max_gap = reentry_max_gap
                max_center_distance = reentry_max_center_distance
                max_area_ratio = cfg.max_reentry_area_ratio
            else:
                max_gap = normal_max_gap
                max_center_distance = normal_max_center_distance
                max_area_ratio = cfg.max_stitch_area_ratio

            if not -max_overlap <= frame_gap <= max_gap:
                continue

            dx = new_stats["first_center_x"] - old_stats["last_center_x"]
            dy = new_stats["first_center_y"] - old_stats["last_center_y"]
            center_distance = (dx**2 + dy**2) ** 0.5

            if center_distance > max_center_distance:
                continue

            old_area = old_stats["mean_box_area"]
            new_area = new_stats["mean_box_area"]

            if old_area <= 0 or new_area <= 0:
                continue

            area_ratio = max(old_area, new_area) / min(old_area, new_area)

            if area_ratio > max_area_ratio:
                continue

            if new_track_id not in best_candidates:
                best_candidates[new_track_id] = (old_track_id, center_distance)
            elif center_distance < best_candidates[new_track_id][1]:
                best_candidates[new_track_id] = (old_track_id, center_distance)

    return {new_track_id: old_track_id for new_track_id, (old_track_id, _) in best_candidates.items()}


def apply_track_stitching(result: VideoTrackingResult, mapping: dict[int, int]) -> VideoTrackingResult:
    """Rewrite track IDs according to a new_track_id -> kept_track_id mapping."""

    def resolve_track_id(track_id: int) -> int:
        seen = set()
        while track_id in mapping and track_id not in seen:
            seen.add(track_id)
            track_id = mapping[track_id]
        return track_id

    detections_with_stitched_tracks = []

    for detection in result.detections:
        track_id = detection.track_id

        if track_id is not None:
            track_id = resolve_track_id(track_id)

        detections_with_stitched_tracks.append(
            VideoTrackDetection(
                frame_id=detection.frame_id,
                timestamp_sec=detection.timestamp_sec,
                track_id=track_id,
                class_id=detection.class_id,
                label=detection.label,
                confidence=detection.confidence,
                xyxy=detection.xyxy,
            )
        )

    return VideoTrackingResult(
        video_path=result.video_path,
        width=result.width,
        height=result.height,
        fps=result.fps,
        detections=detections_with_stitched_tracks,
    )


def deduplicate_track_frame_detections(result: VideoTrackingResult) -> VideoTrackingResult:
    """Keep one highest-confidence detection for each frame and track ID."""

    best_detections = {}

    for detection in result.detections:
        if detection.track_id is None:
            continue

        key = (detection.frame_id, detection.track_id)

        if key not in best_detections:
            best_detections[key] = detection
        elif detection.confidence > best_detections[key].confidence:
            best_detections[key] = detection

    deduplicated_detections = sorted(best_detections.values(), key=lambda detection: (detection.frame_id, detection.track_id))

    return VideoTrackingResult(
        video_path=result.video_path, width=result.width, height=result.height, fps=result.fps, detections=deduplicated_detections
    )


def group_detections_by_track_and_frame(
    result: VideoTrackingResult,
) -> dict[int, dict[int, list[VideoTrackDetection]]]:
    """Index detections by track ID and frame ID for fast overlap checks."""

    grouped: dict[int, dict[int, list[VideoTrackDetection]]] = {}

    for detection in result.detections:
        if detection.track_id is None:
            continue

        grouped.setdefault(detection.track_id, {})
        grouped[detection.track_id].setdefault(detection.frame_id, [])
        grouped[detection.track_id][detection.frame_id].append(detection)

    return grouped


def box_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute intersection-over-union between two xyxy bounding boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def merge_overlapping_duplicate_tracks(
    result: VideoTrackingResult,
    stats: dict[int, dict[str, int | float]],
    cfg: TrackPostProcessingConfig,
) -> dict[int, int]:
    """Find track IDs that overlap in time and likely describe the same object."""

    grouped = group_detections_by_track_and_frame(result)
    mapping: dict[int, int] = {}

    track_ids = list(grouped.keys())

    for i, track_a in enumerate(track_ids):
        for track_b in track_ids[i + 1 :]:
            frames_a = set(grouped[track_a].keys())
            frames_b = set(grouped[track_b].keys())
            common_frames = sorted(frames_a & frames_b)

            if len(common_frames) < cfg.min_duplicate_overlap_frames:
                continue

            ious = []

            for frame_id in common_frames:
                det_a = max(grouped[track_a][frame_id], key=lambda d: d.confidence)
                det_b = max(grouped[track_b][frame_id], key=lambda d: d.confidence)

                ious.append(box_iou(det_a.xyxy, det_b.xyxy))

            mean_iou = sum(ious) / len(ious)

            if mean_iou < cfg.min_duplicate_iou:
                continue

            if stats[track_a]["count"] >= stats[track_b]["count"]:
                keep_track = track_a
                merge_track = track_b
            else:
                keep_track = track_b
                merge_track = track_a

            mapping[merge_track] = keep_track

    return mapping


def compute_active_track_scores(
    result: VideoTrackingResult,
    stats: dict[int, dict[str, int | float]],
) -> dict[int, float]:
    """Rank tracks by how likely they are to be active match players."""

    image_diag = (result.width**2 + result.height**2) ** 0.5
    image_area = result.width * result.height

    scores = {}

    for track_id, track_stats in stats.items():
        presence_ratio = track_stats["presence_ratio"]
        mean_conf = track_stats["mean_conf"]
        path_distance_ratio = track_stats["total_path_distance"] / image_diag
        mean_box_area_ratio = track_stats["mean_box_area"] / image_area

        scores[track_id] = 2.0 * presence_ratio + 1.0 * mean_conf + 3.0 * path_distance_ratio + 1.0 * mean_box_area_ratio

    return scores


def select_active_tracks(
    result: VideoTrackingResult,
    stats: dict[int, dict[str, int | float]],
    cfg: TrackPostProcessingConfig,
) -> tuple[VideoTrackingResult, dict]:
    """Keep only the top-scoring active tracks when max_tracks is configured."""

    scores = compute_active_track_scores(result, stats)

    if cfg.max_tracks is None:
        return result, {
            "active_track_scores": scores,
            "selected_track_ids": None,
            "dropped_track_ids": [],
        }

    tracks_sorted = sorted(scores, key=scores.get, reverse=True)
    selected_track_ids = set(tracks_sorted[: cfg.max_tracks])

    selected_detections = [detection for detection in result.detections if detection.track_id in selected_track_ids]

    dropped_track_ids = sorted(set(scores) - selected_track_ids)

    return VideoTrackingResult(
        video_path=result.video_path,
        width=result.width,
        height=result.height,
        fps=result.fps,
        detections=selected_detections,
    ), {
        "active_track_scores": scores,
        "selected_track_ids": sorted(selected_track_ids),
        "dropped_track_ids": dropped_track_ids,
    }


def postprocess_tracking_result(result: VideoTrackingResult, cfg: TrackPostProcessingConfig) -> tuple[VideoTrackingResult, dict]:
    """Run the first-pass cleanup pipeline for video tracking results."""

    raw_stats = compute_tracking_stats(result)

    filtered_result, filtering_info = filter_short_tracks(result, raw_stats, cfg)

    filtered_stats = compute_tracking_stats(filtered_result)

    overlap_mapping = merge_overlapping_duplicate_tracks(filtered_result, filtered_stats, cfg)
    overlap_merged_result = apply_track_stitching(filtered_result, overlap_mapping)
    overlap_deduplicated_result = deduplicate_track_frame_detections(overlap_merged_result)
    overlap_merged_stats = compute_tracking_stats(overlap_deduplicated_result)

    stitch_mapping = stitch_tracks(overlap_deduplicated_result, overlap_merged_stats, cfg)
    stitched_result = apply_track_stitching(overlap_deduplicated_result, stitch_mapping)

    deduplicated_result = deduplicate_track_frame_detections(stitched_result)

    stats_before_active_selection = compute_tracking_stats(deduplicated_result)

    active_result, active_selection_info = select_active_tracks(
        deduplicated_result,
        stats_before_active_selection,
        cfg,
    )

    final_stats = compute_tracking_stats(active_result)

    postprocessing_info = {
        "raw_stats": raw_stats,
        "filtering": filtering_info,
        "overlap_mapping": overlap_mapping,
        "stitching_mapping": stitch_mapping,
        "active_selection": active_selection_info,
        "final_stats": final_stats,
    }

    return active_result, postprocessing_info
