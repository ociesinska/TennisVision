from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from mlflow.models import infer_signature

from tennisvision.core.data import build_loaders, build_transforms, make_split
from tennisvision.core.engine import fit, make_optimizer
from tennisvision.core.models import (
    build_model,
    freeze_backbone,
    unfreeze_head,
    unfreeze_resnet_layer4,
)
from tennisvision.core.tracking import setup_mlflow
from tennisvision.core.utils import ensure_dir, get_device, now_tag, seed_everything


@dataclass(frozen=True)
class ExperimentConfig:
    image_root: str
    model_name: str
    seed: int = 42

    # data
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False

    # training
    head_epochs: int = 5
    head_lr: float = 1e-3

    finetune: bool = True
    finetune_epochs: int = 8
    finetune_lr: float = 1e-4

    # splits
    test_size: float = 0.1
    val_size: float = 0.1

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


@torch.no_grad()
def predict_logits(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    all_logits = []
    all_y = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        all_logits.append(logits.detach().cpu())
        all_y.append(yb.detach().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_y, dim=0)


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    seed_everything(cfg.seed)
    device = get_device()
    setup_mlflow(experiment_name="TennisVision1", tracking_uri="http://127.0.0.1:8080")

    # local folder for this path's checkpoints
    run_dir = ensure_dir(Path(cfg.artifacts_dir) / "runs" / f"{now_tag()}_{cfg.model_name}_seed{cfg.seed}")
    head_ckpt_path = run_dir / "best_head.pt"
    ft_ckpt_path = run_dir / "best_finetune.pt"

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
    run_name = getattr(cfg, "run_name", f"{cfg.model_name}_seed{cfg.seed}")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_dict(_jsonable(asdict(cfg)), "config.json")
        mlflow.log_dict({str(k): int(v) for k, v in split.class_to_idx.items()}, "labels/class_to_idx.json")
        mlflow.log_dict(idx_to_class, "labels/idx_to_class.json")

        # head-only
        freeze_backbone(model)
        unfreeze_head(model)
        model = model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = make_optimizer(model, lr=cfg.head_lr)

        hist_head = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            n_epochs=cfg.head_epochs,
            ckpt_path=head_ckpt_path,
        )
        log_history(hist_head, prefix="head/")

        if head_ckpt_path.exists():
            mlflow.log_artifact(str(head_ckpt_path), artifact_path="checkpoints")

        # optional fine-tuning
        hist_ft = None
        if cfg.finetune:
            # for resnets we unfreeze layer4
            # for others: TODO later
            unfreeze_resnet_layer4(model)
            optimizer = make_optimizer(model, lr=cfg.finetune_lr)

            hist_ft = fit(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                n_epochs=cfg.finetune_epochs,
                ckpt_path=ft_ckpt_path,
            )

            log_history(hist_ft, prefix="ft/")

            if ft_ckpt_path.exists():
                mlflow.log_artifact(str(ft_ckpt_path), artifact_path="checkpoints")

        # choose checkpoint to log the model from (head vs finetuned)
        best_path = head_ckpt_path
        best_val = getattr(hist_head, "best_val_metric", None)

        if hist_ft is not None and ft_ckpt_path.exists():
            ft_best_val = getattr(hist_ft, "best_val_metric", None)

            if (ft_best_val is not None) and (best_val is None or ft_best_val >= best_val):
                best_path = ft_ckpt_path
                best_val = ft_best_val

        mlflow.log_metric("best/val_metric", float(best_val) if best_val is not None else -1.0)
        mlflow.set_tag("best_checkpoint", best_path.name)

        # reading best checkpoint before log_model
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        # Model log (signature + input_example), already on best weights
        sample_x, _ = next(iter(val_loader))
        sample_x = sample_x[:1].to(device)

        model.eval()
        with torch.no_grad():
            sample_logits = model(sample_x).detach().cpu().numpy()

        input_example = sample_x.detach().cpu().numpy()
        signature = infer_signature(input_example, sample_logits)

        mlflow.pytorch.log_model(model, artifact_path="model", signature=signature, input_example=input_example)

        mlflow.set_tag("run_id", run.info.run_id)

        return {
            "run_dir": str(run_dir),
            "head_best_val_metric": getattr(hist_head, "best_val_metric", None),
            "ft_best_val_metric": getattr(hist_ft, "best_val_metric", None) if hist_ft else None,
            "best_checkpoint": best_path.name,
        }
