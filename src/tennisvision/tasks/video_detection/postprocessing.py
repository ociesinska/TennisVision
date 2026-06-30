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
    max_tracks: int | None = None


def compute_tracking_stats(result: VideoTrackingResult) -> dict[int, dict[str, int | float]]:

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


def stitch_tracks(result: VideoTrackingResult, stats: dict[int, dict[str, int | float]], cfg: TrackPostProcessingConfig) -> dict[int, int]:

    if not result.detections or not stats:
        return {}

    best_candidates: dict[int, tuple[int, float]] = {}

    total_frames = max(d.frame_id for d in result.detections) + 1

    max_gap = min(cfg.max_stitch_frame_gap, int(total_frames * cfg.max_stitch_gap_ratio))

    max_overlap = min(cfg.max_stitch_overlap, int(total_frames * cfg.max_stitch_overlap_ratio))

    image_diag = (result.width**2 + result.height**2) ** 0.5
    max_center_distance = min(cfg.max_stitch_center_distance, image_diag * cfg.max_stitch_center_distance_ratio)
    for old_track_id, old_stats in stats.items():
        for new_track_id, new_stats in stats.items():
            if old_track_id == new_track_id:
                continue

            frame_gap = new_stats["first_frame"] - old_stats["last_frame"]

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

            if area_ratio > cfg.max_stitch_area_ratio:
                continue

            if new_track_id not in best_candidates:
                best_candidates[new_track_id] = (old_track_id, center_distance)
            elif center_distance < best_candidates[new_track_id][1]:
                best_candidates[new_track_id] = (old_track_id, center_distance)

    return {new_track_id: old_track_id for new_track_id, (old_track_id, _) in best_candidates.items()}


def apply_track_stitching(result: VideoTrackingResult, mapping: dict[int, int]) -> VideoTrackingResult:

    detections_with_stitched_tracks = []

    for detection in result.detections:
        track_id = detection.track_id

        if track_id is not None and track_id in mapping:
            track_id = mapping[track_id]

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


def postprocess_tracking_result(result: VideoTrackingResult, cfg: TrackPostProcessingConfig) -> tuple[VideoTrackingResult, dict]:

    raw_stats = compute_tracking_stats(result)

    filtered_result, filtering_info = filter_short_tracks(result, raw_stats, cfg)

    filtered_stats = compute_tracking_stats(filtered_result)

    stitch_mapping = stitch_tracks(filtered_result, filtered_stats, cfg)

    stitched_result = apply_track_stitching(filtered_result, stitch_mapping)

    deduplicated_result = deduplicate_track_frame_detections(stitched_result)

    final_stats = compute_tracking_stats(deduplicated_result)

    postprocessing_info = {
        "raw_stats": raw_stats,
        "filtering": filtering_info,
        "stitching_mapping": stitch_mapping,
        "final_stats": final_stats,
    }

    return deduplicated_result, postprocessing_info
