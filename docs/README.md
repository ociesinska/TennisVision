# TennisVision documentation

Use the task guides to run the current implementation and the experiment records
to understand how configurations were compared and selected. All shell examples
assume the repository root as the working directory.

## Project guides

- [Getting started](getting-started.md): installation, environments, MLflow, and troubleshooting.
- [Architecture](architecture.md): project structure, data flow, and module responsibilities.
- [Development](development.md): tests, CI, and documentation conventions.

## Tasks and experiments

| Task | Current workflow | Experiment record |
| --- | --- | --- |
| Shot classification | [Data, training, evaluation, serving, and explanations](tasks/shot-classification.md) | [Implemented experiment workflow and evidence to record](experiments/shot-classification.md) |
| Player detection | [Image inference, API, and training](tasks/detection.md) | [Dataset iterations, metrics, run IDs, and hard examples](experiments/detection.md) |
| Video tracking | [Video inference, identity cleanup, and API](tasks/video-tracking.md) | [Tracker selection, postprocessing iterations, and clip reviews](experiments/video-tracking.md) |

Detection and tracking records retain the existing experiment history. The shot
classification record describes the implemented workflow; historical run comparisons
have not yet been recorded in the repository.

[Project overview and demos](../README.md) · [Third-party notices](../THIRD_PARTY_NOTICES.md)
