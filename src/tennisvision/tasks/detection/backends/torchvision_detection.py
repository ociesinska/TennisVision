from __future__ import annotations

from pathlib import Path


def run_torchvision_experiment(cfg):
    raise NotImplementedError("TorchVision detection training backend is not implemented yet.")


def load_torchvision_detector(model_path: str | Path, device: str = "auto"):
    raise NotImplementedError("TorchVision detection inference backend is not implemented yet.")


def predict_torchvision_image(model, image_path: str | Path, cfg):
    raise NotImplementedError("TorchVision detection inference backend is not implemented yet.")
