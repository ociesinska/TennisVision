# Video Tracking Experiments

## Current Baseline

| Field | Value |
| --- | --- |
| Detector | yolo11s_img1280_ep30 |
| Tracker | ByteTrack |
| Confidence | 0.5 |
| new_track_thresh | 0.8 |
| track_buffer | 60 |
| Notes | Higher detection confidence removed most ball kid / referee false positives in tested clips. Higher `new_track_thresh` reduced unnecessary creation of new track IDs. |

## Notes

Initial video tracking experiments showed that the detector was able to localize both near and far players reliably. However, the tracker often fragmented the identity of the same player into multiple `track_id`s. This happened even in simple singles clips, where the near player was consistently detected but the assigned ID changed between consecutive parts of the rally.

Increasing detection confidence to `0.5` removed most false positives on ball kids and referees in the tested video clips. This threshold is currently used for video tracking inference, but it should still be checked against harder clips with far, occluded, or low-contrast players.

Both ByteTrack and BoT-SORT were tested on the same video clips. BoT-SORT produced similar ID switches, so the issue was not solved by changing the tracker type alone.

## Qualitative Comparison

| Before threshold tuning | After threshold tuning |
| --- | --- |
| ![Tracking before threshold tuning](assets/video1.gif) | ![Tracking after threshold tuning](assets/video1_det.gif) |

The tuned setup uses a higher detection confidence and a higher `new_track_thresh`. In this example, the player detections remain stable while unnecessary detections and new track creation are reduced.

## Tracker Choice

ByteTrack is used as the current baseline tracker. In the tested tennis clips, BoT-SORT did not produce a clear qualitative improvement over ByteTrack: both trackers localized players similarly and both could still produce ID switches. Since the tracking quality was comparable, ByteTrack is preferred because it is simpler, has fewer moving parts, and is easier to tune/debug for the current stage of the project.

BoT-SORT remains a candidate for later experiments, especially if camera motion, longer occlusions, or ReID-based identity recovery become a larger problem. For now, the main improvement came from detection confidence and `new_track_thresh`, not from changing the tracker family.

The most impactful tracker parameter was `new_track_thresh`. Increasing it to `0.8` significantly reduced unnecessary creation of new track IDs, because the tracker became more conservative when deciding that a detection should start a new track instead of being associated with an existing one.

Additional tests with different `track_buffer` values did not noticeably change the result. This suggests that the main issue was not how long lost tracks were kept in memory, but how easily the tracker created new identities. Therefore, the current baseline uses ByteTrack with a higher `new_track_thresh` and a moderate `track_buffer`.

## Postprocessing Motivation

After testing the current detector and tracker on multiple videos, the remaining errors looked mostly like video-level track selection and identity consistency problems rather than pure detection failures. The detector is able to find tennis players reliably, including players on neighboring courts. This is expected because the detection class is defined broadly as `tennis_player`, not only as the active players on the main court.

Therefore, the next stage is postprocessing: the detector/tracker can return all plausible tennis-player tracks, and a separate video-level step should clean short unstable tracks, repair simple ID fragmentation, and eventually select the main-court players.

## Postprocessing Iteration 1

The first postprocessing iteration adds a conservative pipeline focused on cleanup and simple ID repair:

1. Compute per-track statistics:
   - `count`
   - `first_frame`, `last_frame`
   - `duration_frames`
   - `presence_ratio`, `duration_ratio`
   - `mean_conf`
   - `mean_box_area`
   - `total_path_distance`
   - first/last box center

2. Filter short tracks:
   - very short tracks are removed before stitching;
   - the minimum required count is dynamic: it uses both an absolute minimum and a ratio of total video length.

3. Stitch adjacent track fragments:
   - intended for cases where one track ends and another begins shortly after;
   - matching uses frame gap/overlap, center distance, and box area ratio;
   - this handles simple `old_id -> new_id` fragmentation cases.

4. Deduplicate frame-level conflicts:
   - after stitching, multiple boxes may temporarily share the same `(frame_id, track_id)`;
   - the current rule keeps the highest-confidence box for each `(frame_id, track_id)` pair.

5. Render a postprocessed video:
   - raw JSON is not enough to judge tracking quality;
   - the postprocessing script now writes `tracks_postprocessed.json`, `postprocessing_info.json`, and `video_postprocessed.mp4`.

This iteration intentionally does not yet solve all identity problems. It is a first cleanup pass, not a full player-selection system.

## Postprocessing Iteration 2

The second postprocessing iteration adds overlap-based duplicate track merging. This targets cases where two track IDs exist at the same time and describe the same player, rather than cases where one track ends and another starts later.

Added steps:

1. Group detections by `track_id` and `frame_id`.
2. Compare pairs of tracks that overlap in time.
3. Compute box IoU on shared frames.
4. Merge tracks when they overlap for enough frames and have sufficiently high mean IoU.
5. Deduplicate detections again after merging to keep one box per `(frame_id, track_id)`.

On `video6`, this helped with the observed identity-mixing issue where `player 1` was previously confused with `player 4`. After adding overlap-based duplicate merging, this specific ID conflict was no longer visible in the postprocessed output.

This does not solve every remaining issue. In the same set of qualitative checks, a person on the right side of the frame was still detected and tracked as `player`, even though they were not an active player and barely moved. This is not an overlap-duplicate problem; it requires an additional active-player selection step.

## Postprocessing Iteration 3

The third postprocessing iteration adds active-player scoring and optional `max_tracks` filtering. The goal is to keep the most relevant match players from all valid tennis-player detections returned by the detector/tracker.

Added steps:

1. Compute an active-player score per track using:
   - `presence_ratio`
   - `mean_conf`
   - normalized path distance
   - normalized box area

2. Add optional `max_tracks` filtering:
   - singles clips can be evaluated with `--max-tracks 2`;
   - doubles clips can be evaluated with `--max-tracks 4`;
   - when `max_tracks` is not provided, scores are still logged for diagnosis but no active-player filtering is applied.

3. Log postprocessing summary:
   - `summary_postprocessed.json`
   - `postprocessing_info.json`
   - MLflow metrics matching the original video tracking summary.

This helped remove inactive or irrelevant player-like tracks in several clips. However, a global top-N ranking can fail when the same real player is split into multiple track IDs before active selection. In that case, top-2 can select two fragments of the same player and remove the opponent. This was observed on `video6` before stitching parameters were relaxed.

To address this, the stitching configuration was made tunable from the CLI:

- `--max-stitch-frame-gap`
- `--max-stitch-gap-ratio`
- `--max-stitch-center-distance`
- `--max-stitch-center-distance-ratio`
- `--max-stitch-overlap-ratio`

Using a more permissive stitching setup fixed the re-entry identity issue on `video2` and `video6`:

```bash
--max-stitch-frame-gap 180
--max-stitch-gap-ratio 0.25
--max-stitch-center-distance 700
--max-stitch-center-distance-ratio 0.35
```

With these settings, players that temporarily disappeared from the frame and were later re-detected with a new tracker ID were stitched back to the original identity. This preserved identity continuity after re-entry and made `--max-tracks 2` safer for those clips.

There is still a separate detector/input limitation: when a player becomes heavily clipped near the image border or leaves the frame, the detector may miss them for several frames. Postprocessing can preserve the same identity after the player is detected again, but it does not create missing boxes for frames where no detection exists.

## Current Limitations

The current postprocessing handles short noisy tracks, simple adjacent ID switches, and some overlapping duplicate tracks, but some observed errors require additional logic:

- Missing detections near image borders: heavily clipped players can disappear for a few frames before being detected again.
- Global active-player ranking: `--max-tracks 2` can fail if one real player is split into multiple high-scoring fragments before stitching.
- Perspective-specific selection: top-N scoring works for many clips, but a better strategy may be needed for different camera perspectives.
- Main-court player selection: players on neighboring courts may be valid `tennis_player` detections, but they should not necessarily be selected as active players for the main match.

The next postprocessing iteration should explore spatially balanced active-player selection. For standard back-view tennis videos, this may mean selecting one strong track from the near side and one from the far side of the court. For side-view cameras, a more general spatial-diversity strategy may be needed instead of assuming that image `y` position corresponds to near/far court position.
