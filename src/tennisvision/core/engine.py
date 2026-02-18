import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

from tennisvision.core.utils import get_device

logger = logging.getLogger()


def make_optimizer(model, lr=1e-3, wd=1e-4):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


class AccuracyMeter:
    """Simple replacement for torchmetrics: update(logits, y) + compute()."""

    def __init__(self):
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, y: torch.Tensor):
        preds = logits.argmax(dim=1)
        self.correct += (preds == y).sum().item()
        self.total += y.numel()

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.Tensor(0.0)
        return torch.tensor(self.correct) / torch.tensor(self.total)


def train_one_epoch(model, 
                    optimizer, 
                    loss_fn, 
                    metric, 
                    loader: DataLoader, 
                    device: str,
                    scheduler: LRScheduler | None,
                    scheduler_step_per_batch: bool = False
                    ) -> tuple[float, float]:

    model.train()
    metric.reset()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        if scheduler is not None and scheduler_step_per_batch: # e.g. OneCycleLR / ExponentialLR (?): scheduler.step() after each optimizer.step()
            scheduler.step()

        total_loss += loss.item()
        metric.update(y_pred, y)

    return total_loss / len(loader), metric.compute().item()


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_metric: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_metric: list[float] = field(default_factory=list)
    best_val_metric: float = -1.0
    lr: list[float] = field(default_factory=list)


@torch.no_grad()
def evaluate(model, data_loader, loss_fn, metric, device) -> tuple[float, float]:
    model.eval()
    metric.reset()
    total_loss = 0
    n = 0

    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        n += X_batch.size(0)
        metric.update(y_pred, y_batch)

    return total_loss / n, metric.compute().item()


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn: nn.Module,
    metric=None,
    device: str = "cpu",
    n_epochs: int = 5,
    ckpt_path: str | Path | None = None,
    *,
    scheduler: LRScheduler | None = None,
    scheduler_step_per_batch: bool = False
) -> History:

    if metric is None:
        metric = AccuracyMeter()

    best_val_met = -1.0
    best_epoch = -1

    hist = History()

    ckpt_path = Path(ckpt_path) if ckpt_path is not None else None
    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_metric = train_one_epoch(model, optimizer, loss_fn, metric, train_loader, device, scheduler=scheduler,
            scheduler_step_per_batch=scheduler_step_per_batch)
        val_loss, val_metric = evaluate(model, val_loader, loss_fn, metric, device)

        hist.train_loss.append(tr_loss)
        hist.train_metric.append(tr_metric)
        hist.val_loss.append(val_loss)
        hist.val_metric.append(val_metric)
        hist.lr.append(float(optimizer.param_groups[0]["lr"]))

        # lr scheduler per epoch
        if scheduler is not None and not scheduler_step_per_batch:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metric)
            else:
                scheduler.step()

        if val_metric > best_val_met:
            best_val_met = val_metric
            best_epoch = epoch
            hist.best_val_metric = best_val_met
            hist.best_epoch = best_epoch

            # checkpoint - saving only when the results are better
            if ckpt_path is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_val_acc": best_val_met,
                    },
                    ckpt_path,
                )
        logger.info(
            f"Epoch {epoch}/{n_epochs} | "
            f"lr {hist.lr[-1]:.3e}"
            f"train loss {tr_loss:.4f} acc {tr_metric:.4f} | "
            f"val loss {val_loss:.4f} acc {val_metric:.4f}"
        )

    return hist


@dataclass
class Predictions:
    y_true: torch.Tensor | None 
    y_pred: torch.Tensor
    logits: torch.Tensor
    probs: torch.Tensor


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader) -> Predictions:
    
    model.eval()
    device = get_device()

    all_logits = []
    all_y = []

    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            xb, yb = batch
            all_y.append(yb.cpu())
        else:
            xb = batch
        
        xb = xb.to(device)
        logits = model(xb)
        all_logits.append(logits.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=1)
    y_pred = probs.argmax(dim=1)

    y_true = torch.cat(all_y, dim=0) if all_y else None

    return Predictions(y_true=y_true, y_pred=y_pred, probs=probs, logits=logits)


def predict_tensor(model, x: torch.Tensor):
    """Predict a single tensor."""
    model.eval()
    logits = model(x)
    probs=torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    return pred.cpu(), probs.cpu()


def evaluate_and_log_split(
        model: torch.nn.Module, 
        loader: torch.utils.data.DataLoader,
        split_name: str,
        class_names: list[str]) -> None:
    
    preds = predict_loader(model, loader)
    y_true = preds.y_true.numpy()
    y_pred = preds.y_pred.numpy()

    acc = (y_true == y_pred).mean()
    mlflow.log_metric(f"{split_name} / acc", float(acc))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6,6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    mlflow.log_figure(fig, f"{split_name}/confusion_matrix.png")
    plt.close(fig)