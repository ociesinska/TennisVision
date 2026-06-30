# Detection Experiments

## Dataset

| Version | Date | Splits | Notes |
| --- | --- | --- | --- |
| detection_cvat_v1 | YYYY-MM-DD | train / val / test | Corrected player boxes from CVAT. |
| detection_cvat_v1_updated | 2026-06-19 | train / val / test | Same `data/detection` dataset path, extended with additional train/val examples containing ball kids / staff and corrected player boxes. |

## Current Baseline

| Field | Value |
| --- | --- |
| Model | yolo11s.pt |
| Run name | yolo11s_img1280_ep50_playersv2 |
| Dataset | detection_cvat_v1_updated |
| Training run | d21d335020d14f4498d5388aebebd03b |
| Evaluation run | f0f3f35d6d8544a983a6922816c6dc75 |
| mAP50 | 0.9427 |
| mAP50-95 | 0.7208 |
| Precision | 0.9421 |
| Recall | 0.8625 |
| Notes | Current practical baseline: selected from the updated dataset iteration because it includes harder ball kid / staff cases. |

The previous `yolo11s_img1280_ep30` model still has the highest mAP50-95, but it was trained on the earlier dataset version. The current baseline is selected from the updated dataset iteration because it better reflects the target failure cases observed during video tracking and manual inspection.

The current detector is accepted as a practical baseline, not as a final detector. Further detector improvements will be driven by observed video-level failure cases rather than isolated metric optimization.

## Experiment Table

| Run | Model | imgsz | Epochs | Batch | mAP50 | mAP50-95 | Precision | Recall | Takeaway |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| yolo11n_img960_ep30 | yolo11n.pt | 960 | 30 | 4 | 0.9296 | 0.7144 | 0.9660 | 0.8485 | High precision, but lower recall suggests some players are missed. |
| yolo11s_baseline_20260617_183029 | yolo11s.pt | 960 | 30 | 4 | 0.9301 | 0.7243 | 0.9278 | 0.8636 | Better overall than 11n at 960; recall improves with more capacity. |
| yolo11m_img960_ep30 | yolo11m.pt | 960 | 30 | 4 | 0.9009 | 0.6913 | 0.8996 | 0.8561 | Larger model did not help; likely too heavy for dataset size or noisier boxes. |
| yolo11s_img1280_ep30 | yolo11s.pt | 1280 | 30 | 2 | 0.9240 | 0.7443 | 0.8922 | 0.8778 | Best so far; higher image size likely helps with small/far players. |
| yolo11n_img1280_ep30 | yolo11n.pt | 1280 | 30 | 2 | 0.8979 | 0.6589 | 0.8623 | 0.8542 | Higher image size hurt 11n; the smallest model likely cannot use the extra detail effectively. |
| yolo11s_img960_ep50 | yolo11s.pt | 960 | 50 | 4 | 0.9124 | 0.7233 | 0.9185 | 0.8543 | Extra epochs at 960 did not beat 11s at 1280. |
| yolo11s_img1280_ep30_playersv2 | yolo11s.pt | 1280 | 30 | 2 | 0.9349 | 0.7203 | 0.9398 | 0.8561 | Updated train/val data improves mAP50 and precision, but lowers mAP50-95 and recall; needs hard-example visual regression check. |
| yolo11s_img1280_ep40_playersv2 | yolo11s.pt | 1280 | 40 | 2 | 0.9243 | 0.7162 | 0.8973 | 0.8788 | Intermediate training length improves recall, but does not beat the ep30/ep50 updated-data runs on mAP50 or precision. |
| yolo11s_img1280_ep50_playersv2 | yolo11s.pt | 1280 | 50 | 2 | 0.9427 | 0.7208 | 0.9421 | 0.8625 | Longer training on updated data improves mAP50/precision slightly, but still does not recover baseline mAP50-95. |

## Run IDs

| Run | Train run ID | Eval run ID | Notes |
| --- | --- | --- | --- |
| yolo11n_img960_ep30 | 7af28936c39942d0957b3c17ac090ec4 | 3f68f824a3b5474b9a4e10b4507b881b | mAP75: 0.8111, fitness: 0.7144 |
| yolo11s_baseline_20260617_183029 | 9c5fab84168e4ef285c40e0407f8eb7f | a062d019565c4a94a251511686772e2f | mAP75: 0.7940, fitness: 0.7243 |
| yolo11m_img960_ep30 | 5a212fa0cd7d494d8926d2b0c734bc71 | bea2e88524744d3aabefe22cedc387ad | mAP75: 0.7311, fitness: 0.6913 |
| yolo11s_img1280_ep30 | 50bbeedc56394af7b96c42947f15d5f7 | 82950eb7b17d4507b12dad5f59ed746a | mAP75: 0.8249, fitness: 0.7443|
| yolo11n_img1280_ep30 | c8e926040e404eb09d1b4755ad1ca90c | 3aa2712a9266435fbfbde6df76ad6e98 | mAP75: 0.7074, fitness: 0.6589|
| yolo11s_img960_ep50 | cf240eb183af4d28b7595b0589f088a8 | 75e2a4bca11248c789851ecb4bde7f21 | mAP75: 0.8086, fitness: 0.7233|
| yolo11s_img1280_ep30_playersv2 | 3aa861f3fe824d65a6ba4b996ac66318 | 12ef4da34a98484d86a7107782cce531 | mAP75: 0.7466, fitness: 0.7203|
| yolo11s_img1280_ep40_playersv2 | c5bd48422ecb4955b882b9aad04ac553 | 8cba71d067d542acaa374aa09ba6b3db | mAP75: 0.7776, fitness: 0.7162 |
| yolo11s_img1280_ep50_playersv2 | d21d335020d14f4498d5388aebebd03b | f0f3f35d6d8544a983a6922816c6dc75 | mAP75: 0.7608, fitness: 0.7208 |

## Hard Examples

| Image | Split | Model | Problem | Likely Cause | Next Action |
| --- | --- | --- | --- | --- | --- |
| local improvement images | val / improvement set | yolo11s_img1280_ep30 | Ball kids / staff near the back of the court are detected as `player` together with the real player. | Current dataset has too few hard negatives with non-player people on court; visually, ball kids and referees look similar to distant players. | Add more hard negative frames from match videos and annotate only actual players; keep ball kids, referees, and staff unlabelled unless moving to a multi-class setup. |

### Ball Kids / Staff False Positives

Observed failure mode: ball kids / staff visible on or near the court are detected as `player`. This is especially visible for small people close to the back wall, side benches, umpire chair, or court edges. The next dataset iteration should include more hard negative frames from match videos, with annotations kept only on actual players.

Important dataset balance: adding many ball kid / staff negatives may make the model too restrictive around court edges and background areas. The next dataset version should also add hard positive examples where real players are close to the net, close to court boundaries, partially outside the main court area, or visually similar to staff positions. This should help the model learn the difference between role/context and actual player appearance instead of simply suppressing people near the edges.

Local examples to revisit:

| Local file | Reason | Expected behavior |
| --- | --- | --- |
| `data/images_to_improve_player_detection/raw/img_imp1.png` | ball kids / staff behind the court | detect actual players only |
| `data/images_to_improve_player_detection/raw/img_imp2.png` | staff visible near court area | ignore non-player people |
| `data/images_to_improve_player_detection/raw/img_imp3.png` | ball kids visible on or near court | ignore non-player people |

## Repro Commands

### Train Current Baseline

```bash
uv run python -m tennisvision.tasks.detection.scripts.run_experiment \
  --model yolo11s.pt \
  --run-name yolo11s_img1280_ep30 \
  --epochs 30 \
  --imgsz 1280 \
  --batch 2
```

### Evaluate Current Baseline

```bash
uv run python -m tennisvision.tasks.detection.scripts.evaluate \
  --run-id 50bbeedc56394af7b96c42947f15d5f7 \
  --model-artifact-path weights/best.pt \
  --data-config data/detection/data.yaml \
  --split test \
  --dataset-tag detection_test_cvat_v1 \
  --model-tag yolo11s_img1280_ep30
```

### Check Hard Example

```bash
uv run python -m tennisvision.tasks.detection.scripts.infer \
  --image data/images_to_improve_player_detection/raw/img_imp1.png \
  --run-id 50bbeedc56394af7b96c42947f15d5f7 \
  --model-artifact-path weights/best.pt \
  --visualize true
```
