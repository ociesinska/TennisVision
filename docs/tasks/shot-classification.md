# Shot classification

[Documentation index](../README.md) · [Experiment workflow](../experiments/shot-classification.md)

This task classifies individual tennis images. The serving fallback labels are
`backhand`, `forehand`, `ready_position`, and `serve`; training derives its labels
from dataset folders and logs the mapping with the model. It is not a temporal
shot-event detector.

## Data and preprocessing

The default dataset location is `data/Tennis positions/images`. Supply an ImageFolder
directory with one subdirectory per class, for example:

```text
images/
├── backhand/
├── forehand/
├── ready_position/
└── serve/
```

[`data.py`](../../src/tennisvision/tasks/shot_classification/data.py) creates a seeded,
stratified image-level split: 80% training, 10% validation, 10% test by default.
Split indices and labels are saved to MLflow. Reusing those indices requires the
same dataset contents and ImageFolder ordering. There is no match/video grouping
in the current split, so related frames are not automatically kept together.

Training augmentation uses random resized crops, horizontal flips, and color jitter.
Validation during training uses the pretrained weights' transforms. The evaluation
CLI and API currently use a 224×224 resize with ImageNet normalization; this is not
necessarily identical to every architecture's pretrained validation transform.
Keep that distinction in mind when comparing training and serving results.

## Train

Install the project and start MLflow using the [setup guide](../getting-started.md).
Then run:

```bash
uv run python -m tennisvision.tasks.shot_classification.scripts.run_experiment \
  --image_root "data/Tennis positions/images" \
  --model_name resnet18 \
  --head_epochs 5 \
  --finetune_epochs 8 \
  --seed 42
```

Supported architectures are `resnet18`, `resnet50`, `efficientnet_b0`,
`mobilenet_v3_large`, and `convnext_tiny`. Training starts with pretrained weights,
freezes the backbone, and trains the classification head. The optional second
stage currently unfreezes `layer4` for ResNet models. For the other architectures,
the current helper does not unfreeze additional backbone layers.

The optimizer is AdamW with cross-entropy loss. Configuration supports label
smoothing, weight decay, a validation-driven learning-rate scheduler, and early
stopping. The CLI exposes only the arguments shown above; additional options are
in [`ExperimentConfig`](../../src/tennisvision/tasks/shot_classification/experiments.py).
Training defaults to the `TennisVision` MLflow experiment at
`http://127.0.0.1:8080`; that CLI does not expose a tracking-URI flag.

Outputs include `config.json`, `split/indices.json`, `labels/idx_to_class.json`,
per-epoch metrics, and a model artifact at `model`. With checkpoint saving enabled,
local files are written under `artifacts/runs/<timestamp>_<model>_seed<seed>/`, with
`best_head.pt` and, when applicable, `best_finetune.pt`. Model selection compares the
best validation accuracy of the two phases. Test evaluation is a separate step.

## Evaluate a saved run

```bash
uv run python -m tennisvision.tasks.shot_classification.scripts.infer \
  --run_id YOUR_TRAINING_RUN_ID \
  --image_root "data/Tennis positions/images"
```

Despite its name, this script evaluates the saved test split, rather than taking a
single image. It loads the source run's split indices and label mapping, computes
accuracy and per-class metrics, and logs a confusion matrix and misclassified
examples to `TennisVisionInference`. It also accepts `--model_alias` with a registered
`--model_name` instead of `--run_id`. Use the API below for individual images.

## Hyperparameter search

```bash
uv run python -m tennisvision.tasks.shot_classification.scripts.tune_hpo \
  --model_name mobilenet_v3_large \
  --image_root "data/Tennis positions/images" \
  --trials 15
```

The current Optuna entry point uses four head-training epochs, disables fine-tuning,
and searches weight decay, label smoothing, and scheduler settings. It maximizes a
validation metric and reruns the selected configuration with model/checkpoint and
explainability output. The final run retains the entry point's head-only configuration.
This command describes the implemented search, not a recorded winning experiment.

## Serve images

The API currently reads `MODEL_URI`, `RUN_ID`, `MODEL_NAME`, `DEVICE`, and
`MLFLOW_TRACKING_URI`. The task-prefixed names in the settings module are not yet
wired into this API. To load a training run explicitly:

```bash
export MODEL_URI=""
export RUN_ID="YOUR_TRAINING_RUN_ID"
export MODEL_NAME="resnet18"
uv run uvicorn tennisvision.tasks.shot_classification.api.app:app --port 8000 --reload
```

`MODEL_NAME` identifies the architecture for Grad-CAM. `MODEL_URI` takes precedence
over `RUN_ID` and defaults to `models:/TennisVision@champion`, so clear it when using
a run ID. Using that default URI requires an existing registered model and alias.
Labels come from `IDX_TO_CLASS` JSON, then the run artifact when available, then the
four-class fallback. Ensure the mapping matches the selected model.

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@image.jpg"
curl -X POST "http://localhost:8000/predict_batch?top_k=3" \
  -F "files=@image1.jpg" -F "files=@image2.jpg"
curl -X POST "http://localhost:8000/explain" \
  -F "file=@image.jpg" --output explanation.png
```

`/health` reports readiness; `/predict` returns a label, confidence, and top-k
probabilities. `/explain` returns a PNG with Grad-CAM overlays for the top two classes.
The API limits single-image payloads to 10 MB and batches to 64 files / 25 MB.

## Generate explanations from the CLI

```bash
uv run python -m tennisvision.tasks.shot_classification.scripts.explain \
  --run_id YOUR_TRAINING_RUN_ID --model_name resnet18 --image image.jpg
```

Use `--input-dir` instead of `--image` for a directory. The script logs Grad-CAM
overlays to MLflow. Specify the architecture matching the model; an attention
visualization helps inspect predictions but does not replace held-out evaluation.
