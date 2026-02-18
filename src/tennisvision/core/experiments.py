from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from mlflow.models import infer_signature
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tennisvision.core.data import build_loaders, build_transforms, make_split
from tennisvision.core.engine import evaluate_and_log_split, fit, make_optimizer
from tennisvision.core.mlflow_utils import setup_mlflow
from tennisvision.core.models import (
    build_model,
    freeze_backbone,
    unfreeze_head,
    unfreeze_resnet_layer4,
)
from tennisvision.core.utils import ensure_dir, get_device, now_tag, seed_everything

logger = logging.getLogger()

@dataclass(frozen=True)
class ExperimentConfig:
    image_root: str
    model_name: str
    seed: int = 42

    # data
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True

    # training
    head_epochs: int = 5
    head_lr: float = 1e-3

    use_scheduler: bool = True
    head_scheduler_patience: int = 2
    head_scheduler_factor: float = 0.5

    finetune: bool = True
    finetune_epochs: int = 8
    finetune_lr: float = 1e-4
    ft_scheduler_patience: int = 2
    ft_scheduler_factor: float = 0.5

    # splits
    test_size: float = 0.1
    val_size: float = 0.1

    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    
    run_name: str | None = None

    mlflow_experiment_name: str = "TennisVision"
    mlflow_tracking_uri: str = "http://127.0.0.1:8080"

    artifacts_dir: Path = Path("artifacts")
    mlflow_dir: Path = Path("artifacts/mlflow")


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    """Changes Path/torch.device etc. to serialized values to JSON"""
    out: dict[str, Any] = {}

    for k, v in d.items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def log_history(history: Any, prefix: str = "") -> None:

    if isinstance(history, dict):
        hist = history
    else:
        hist = history.__dict__

    # logging metrics with step = epoch :contentReference[oaicite:6]{index=6}

    n = max((len(v) for v in hist.values() if isinstance(v, list)), default=0)

    for epoch in range(n):
        for k, v in hist.items():
            if isinstance(v, list) and epoch < len(v):
                key = f"{prefix}{k}" if prefix else k
                mlflow.log_metric(key, float(v[epoch]), step=epoch + 1)  # :contentReference[oaicite:7]{index=7}

def run_experiment(
        cfg: ExperimentConfig,
        *,
        log_model: bool = True,
        log_confusion_matrix: bool = True,
        save_checkpoints: bool = True
        ) -> dict[str, Any]:
    
    seed_everything(cfg.seed)
    device = get_device()

    has_active_run = mlflow.active_run() is not None

    setup_mlflow(experiment_name=cfg.mlflow_experiment_name,
                 tracking_uri=cfg.mlflow_tracking_uri,
                 set_experiment= not has_active_run)

    # local folder for this path's checkpoints

    run_dir = None
    head_ckpt_path = None
    ft_ckpt_path = None

    if save_checkpoints:
        run_dir = ensure_dir(Path(cfg.artifacts_dir) / "runs" / f"{now_tag()}_{cfg.model_name}_seed{cfg.seed}")
        head_ckpt_path = run_dir / "best_head.pt"
        ft_ckpt_path = run_dir / "best_finetune.pt"

    logger.info(f"Fitting pretrained model {cfg.model_name}")
    # split
    split = make_split(cfg.image_root, seed=cfg.seed, test_size=cfg.test_size, val_size=cfg.val_size)

    # model & weights
    num_classes = len(split.class_to_idx)

    model, weights = build_model(cfg.model_name, num_classes=num_classes, pretrained=True)

    #   # 3) transforms + loaders

    train_tfms, val_tfms = build_transforms(weights, train_aug=True)
    (train_loader, val_loader, test_loader), (_, _, _), classes = build_loaders(
        cfg.image_root,
        split,
        train_tfms,
        val_tfms,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    # label mapping (JSON keys must be str)
    idx_to_class = {str(v): k for k, v in split.class_to_idx.items()}

    run_name = cfg.run_name or f"{cfg.model_name}_seed{cfg.seed}"

    
    with mlflow.start_run(run_name=run_name, nested=has_active_run) as run:
        mlflow.log_dict(_jsonable(asdict(cfg)), "config.json")
        mlflow.log_dict(idx_to_class, "labels/idx_to_class.json")

        # head-only
        freeze_backbone(model)
        unfreeze_head(model)
        model = model.to(device)
        scheduler_head = None

        loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        optimizer = make_optimizer(model, lr=cfg.head_lr, wd = cfg.weight_decay)
        logger.info("Fitting head of the pretrained model.")
        
        if cfg.use_scheduler:
            scheduler_head = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor = cfg.head_scheduler_factor,
                patience = cfg.head_scheduler_patience,
            )

        hist_head = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            n_epochs=cfg.head_epochs,
            ckpt_path=head_ckpt_path,
            scheduler=scheduler_head,
            scheduler_step_per_batch=False
        )

        log_history(hist_head, prefix="head/")

        if head_ckpt_path and head_ckpt_path.exists():
            mlflow.log_artifact(str(head_ckpt_path), artifact_path="checkpoints")

        # optional fine-tuning
        hist_ft = None

        if cfg.finetune:
            logger.info("Fine tuning")
            # for resnets we unfreeze layer4
            # for others: TODO later
            unfreeze_resnet_layer4(model)
            optimizer = make_optimizer(model, lr=cfg.finetune_lr, wd=cfg.weight_decay)
            
            scheduler_ft = None
            if cfg.use_scheduler:
                scheduler_ft = ReduceLROnPlateau(
                    optimizer,
                    mode="max", # max accuracy
                    factor=cfg.ft_scheduler_factor,
                    patience=cfg.ft_scheduler_patience,
                    min_lr=1e-6
                )

            hist_ft = fit(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                n_epochs=cfg.finetune_epochs,
                ckpt_path=ft_ckpt_path,
                scheduler=scheduler_ft,
                scheduler_step_per_batch=False
            )

            log_history(hist_ft, prefix="ft/")

            if ft_ckpt_path and ft_ckpt_path.exists():
                mlflow.log_artifact(str(ft_ckpt_path), artifact_path="checkpoints")

        # choose checkpoint to log the model from (head vs finetuned)
        best_path = head_ckpt_path
        best_val = getattr(hist_head, "best_val_metric", None)

        if hist_ft is not None and ft_ckpt_path and ft_ckpt_path.exists():
            ft_best_val = getattr(hist_ft, "best_val_metric", None)

            if (ft_best_val is not None) and (best_val is None or ft_best_val >= best_val):
                best_path = ft_ckpt_path
                best_val = ft_best_val

        mlflow.log_metric("best/val_metric", float(best_val) if best_val is not None else -1.0)
        if best_path:
            mlflow.set_tag("best_checkpoint", best_path.name)

        # reading best checkpoint before log_model
        if best_path and best_path.exists():
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])


        # inference and evaluation on validation set
        if log_confusion_matrix:
            evaluate_and_log_split(model, val_loader, "val", class_names=classes)

        # Model log (signature + input_example), already on best weights
        sample_x, _ = next(iter(val_loader))
        sample_x = sample_x[:1].to(device)

        model.eval()
        with torch.no_grad():
            sample_logits = model(sample_x).detach().cpu().numpy()

        input_example = sample_x.detach().cpu().numpy()
        signature = infer_signature(input_example, sample_logits)

        if log_model:
            mlflow.pytorch.log_model(model, artifact_path="model", signature=signature, input_example=input_example)

        mlflow.set_tag("run_id", run.info.run_id)

        return {
            "run_dir": str(run_dir),
            "head_best_val_metric": getattr(hist_head, "best_val_metric", None),
            "ft_best_val_metric": getattr(hist_ft, "best_val_metric", None) if hist_ft else None,
            "best_checkpoint": best_path.name if best_path and best_path.exists() else None,
        }

def load_mlflow_model_from_run(
        model,
        run_id: str,
        # tracking_uri: str = "http://127.0.0.1:8080",
        model_artifact_path: str = "model"
) -> torch.nn.Module:
    """Load model from Mlflow for run_id."""
    setup_mlflow()
    device = get_device()
    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    model = model.pytorch.load_model(model_uri)
    model.to(device)
    return model.to(device)


def load_mlflow_model_from_registry(
        model_uri: str,
        # tracking_uri: str,
):
    """Load Mlflow model from registry"""
    setup_mlflow()
    model = mlflow.pytorch.load_model(model_uri)
    device = get_device()
    model.to(device)

    return model
