import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


def train_one_epoch(model, optimizer, loss_fn, metric, loader: DataLoader, device: str) -> tuple[float, float]:

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
    best_epoch: int = -1


@torch.no_grad
def evaluate(model, data_loader, loss_fn, metric, device) -> tuple[float, float]:
    model.eval()
    metric.reset()
    total_loss = 0

    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        total_loss += loss.item()
        metric.update(y_pred, y_batch)

    return total_loss / len(data_loader), metric.compute().item()


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
        tr_loss, tr_metric = train_one_epoch(model, optimizer, loss_fn, metric, train_loader, device)
        val_loss, val_metric = evaluate(model, val_loader, loss_fn, metric, device)

        hist.train_loss.append(tr_loss)
        hist.train_metric.append(tr_metric)
        hist.val_loss.append(val_loss)
        hist.val_metric.append(val_metric)

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
            f"train loss {tr_loss:.4f} acc {tr_metric:.4f} | "
            f"val loss {val_loss:.4f} acc {val_metric:.4f}"
        )

    return hist
