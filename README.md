# TennisVision 🎾

TennisVision is a modular computer vision project for tennis analytics, featuring model training, inference, experiment tracking, and a REST API for tennis shot classification.

## Features

- **Model Training & Experimentation**: Train and tune models for tennis shot classification.
- **Batch & Single Prediction API**: FastAPI endpoints for single and batch image inference.
- **MLflow Integration**: Track experiments, manage models, and load artifacts via MLflow.
- **Data Processing & Visualization**: Utilities for preprocessing, augmentation, and result visualization.
- **Script Automation**: CLI scripts for inference, experiment runs, alias management, and hyperparameter tuning.

## Installation

1. Create a virtual environment and activate it:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install in editable mode:
   ```bash
   pip install -e .
   ```

## Running the MLflow & FastAPI Servers

1. Start MLflow server (required for model loading):
   ```bash
   mlflow ui --port 8080
   ```
2. Start FastAPI server:
   ```bash
   uv run uvicorn tennisvision.api.app:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Endpoints

- `POST /predict`: Predict tennis shot class for a single image (multipart/form-data).
- `POST /predict_batch`: Predict classes for multiple images in a batch (multipart/form-data).
- `GET /health`: Health check endpoint.

### Example Usage

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@image.jpg"
```

## Core Modules

- **src/tennisvision/api/app.py**: FastAPI app, endpoints, model loading, batch/single prediction logic.
- **src/tennisvision/core/**:
  - `data.py`: Data loading and preprocessing utilities.
  - `engine.py`: Training and evaluation engine.
  - `experiments.py`: Experiment orchestration and tracking.
  - `mlflow_utils.py`: MLflow model loading and artifact management.
  - `models.py`: Model definitions (e.g., ResNet, EfficientNet, MobileNet).
  - `utils.py`: Helper functions for metrics, logging, etc.
  - `viz.py`: Visualization tools for results and data.
- **src/tennisvision/scripts/**:
  - `infer.py`: CLI for running inference on images or folders.
  - `run_experiment.py`: Script for launching training experiments.
  - `set_alias.py`: Manage experiment aliases.
  - `tune_hpo.py`: Hyperparameter optimization script.

## Data & Artifacts

- **data/**: Raw and processed tennis position datasets.
- **artifacts/**: MLflow artifacts and experiment outputs.
- **mlartifacts/**, **mlruns/**: MLflow model registry and run tracking.

## Troubleshooting

- If you get errors about MLflow, make sure the MLflow server is running on port 8080.
- To kill a process on port 8000:
  ```bash
  lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9
  ```
- For batch prediction errors, check tensor conversion and model loading logs.

## Project Structure

```
TennisVision/
├── src/
│   └── tennisvision/
│       ├── api/
│       │   └── app.py
│       ├── core/
│       │   ├── data.py
│       │   ├── engine.py
│       │   ├── experiments.py
│       │   ├── mlflow_utils.py
│       │   ├── models.py
│       │   ├── utils.py
│       │   └── viz.py
│       ├── scripts/                # General scripts for all tasks
│       │   ├── infer.py
│       │   ├── run_experiment.py
│       │   ├── set_alias.py
│       │   └── tune_hpo.py
│       └── tasks/
│           ├── shot_classification/
│           │   └── scripts/        # Task-specific scripts for shot classification
│           ├── court_keypoints/
│           │   └── scripts/        # Task-specific scripts for court keypoints (future)
│           └── ...                 # Other tasks (future)
├── data/
├── artifacts/
├── mlartifacts/
├── mlruns/
├── README.md
├── pyproject.toml
└── Makefile
```
