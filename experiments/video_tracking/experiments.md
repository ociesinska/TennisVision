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
