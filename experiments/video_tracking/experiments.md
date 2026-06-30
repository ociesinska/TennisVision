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

## Current Limitations

The current postprocessing handles short noisy tracks and simple adjacent ID switches, but some observed errors require additional logic:

- Long-gap re-entry: a player disappears from the frame for a longer time and returns with a new ID. This is not well handled by adjacent stitching because the frame gap and spatial distance can be large.
- Overlapping duplicate tracks: two IDs can exist at the same time for what appears to be the same player. This requires comparing detections frame-by-frame during the overlap period, not only comparing first/last track statistics.
- Main-court player selection: players on neighboring courts may be valid `tennis_player` detections, but they should not necessarily be selected as active players for the main match.

The next postprocessing iteration should add overlapping-duplicate merging first, then a track scoring / main-court selection step.
