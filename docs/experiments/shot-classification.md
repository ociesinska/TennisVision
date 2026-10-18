# Shot-classification experiment workflow

[Documentation index](../README.md) · [Usage guide](../tasks/shot-classification.md)

There are no recorded run comparisons, scores, or selected baseline for this task
in the repository documentation yet. The following describes the workflow implemented
in the code, not a reconstructed chronology of experiments.

## Implemented workflow

| Stage | Purpose | Evidence produced |
| --- | --- | --- |
| Seeded stratified split | Keep training/validation/test assignments reproducible for a fixed dataset | `split/indices.json`, label mapping |
| Pretrained head training | Adapt a classifier to the image classes | Head metrics and `best_head.pt` |
| Optional ResNet fine-tuning | Adapt the final backbone block | Fine-tuning metrics and `best_finetune.pt` |
| Validation-based selection | Choose between saved head and fine-tuned checkpoints | `best/val_metric`, checkpoint tag, logged model |
| Optuna search | Compare regularization/scheduler configurations using validation performance | Trial parameters, validation objectives, final selected run |
| Test evaluation | Assess a saved model on its held-out image split | Accuracy, per-class F1/accuracy, confusion matrix, misclassified examples |
| Grad-CAM inspection | Examine spatial evidence behind predictions | Top-two-class overlays |

Commands and current configuration details are in the [task guide](../tasks/shot-classification.md).
Supported architectures are implementation options, not evidence that each was
trained or compared successfully.

## Recording the next iteration

For each completed comparison, add the question being tested, dataset version and
split seed, architecture, head/fine-tuning settings, training/evaluation run IDs,
validation and test metrics, representative mistakes, and the resulting decision.
Keep validation-based model selection separate from reporting test performance.

Before interpreting results, record whether related images originate from the same
video/match and whether evaluation preprocessing matches the trained model. The
current split is image-level and the current serving/evaluation resize is fixed;
these details affect what a comparison demonstrates.

This page is the canonical home for future shot-classification results. Add measured
entries when runs are available rather than duplicating result tables in the README.
