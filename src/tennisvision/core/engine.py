import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

from tennisvision.core.explainability import explainability_for_training

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


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.best_val_loss = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss):
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed for {self.patience} epochs.")


def train_one_epoch(
    model, optimizer, loss_fn, metric, loader: DataLoader, device: str, scheduler: LRScheduler | None, scheduler_step_per_batch: bool = False
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

        if scheduler is not None and scheduler_step_per_batch:  # e.g. OneCycleLR / ExponentialLR (?): scheduler.step() after each optimizer.step()
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
    scheduler_step_per_batch: bool = False,
    explainability: bool = False
) -> History:

    if metric is None:
        metric = AccuracyMeter()

    best_val_met = -1.0
    best_epoch = -1

    hist = History()
    early_stopping = EarlyStopping()

    ckpt_path = Path(ckpt_path) if ckpt_path is not None else None
    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    explain_images = {}

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_metric = train_one_epoch(
            model, optimizer, loss_fn, metric, train_loader, device, scheduler=scheduler, scheduler_step_per_batch=scheduler_step_per_batch
        )
        val_loss, val_metric = evaluate(model, val_loader, loss_fn, metric, device)

        hist.train_loss.append(tr_loss)
        hist.train_metric.append(tr_metric)
        hist.val_loss.append(val_loss)
        hist.val_metric.append(val_metric)
        hist.lr.append(float(optimizer.param_groups[0]["lr"]))

        if explainability:
            img1, img2 = explainability_for_training(model, epoch, train_loader, device, explain_every=2, explain_sample = 3)
            if img1 is not None and img2 is not None:
                explain_images[epoch] = img1, img2

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

        early_stopping.check_early_stop(val_loss)
        if early_stopping.stop_training:
            logger.info(f"Early stopping at epoch {epoch}")
            break

        logger.info(
            f"Epoch {epoch}/{n_epochs} | "
            f"lr {hist.lr[-1]:.3e}"
            f"train loss {tr_loss:.4f} acc {tr_metric:.4f} | "
            f"val loss {val_loss:.4f} acc {val_metric:.4f}"
        )

    return hist, explain_images

@dataclass
class Predictions:
    y_true: torch.Tensor | None
    y_pred: torch.Tensor
    logits: torch.Tensor | None = None
    probs: torch.Tensor | None = None


@torch.inference_mode()
def predict_loader(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, *, return_logits: bool = True, return_probs: bool = True
) -> Predictions:

    model.eval()
    model.to(device)

    y_true_chunks = []
    y_pred_chunks = []
    logits_chunks = [] if return_logits else None
    probs_chunks = [] if return_probs else None

    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            xb, yb = batch
            y_true_chunks.append(yb.cpu())
        else:
            xb = batch

        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds = logits.argmax(dim=1).detach().cpu()
        y_pred_chunks.append(preds)

        if return_logits:
            logits_chunks.append(logits.detach().cpu())

        if return_probs:
            probs = torch.softmax(logits, dim=1).detach().cpu()
            probs_chunks.append(probs)

    y_true = torch.cat(y_true_chunks, dim=0) if y_true_chunks else None
    y_pred = torch.cat(y_pred_chunks, dim=0)

    out_logits = torch.cat(logits_chunks, dim=0) if return_logits else None
    out_probs = torch.cat(probs_chunks, dim=0) if return_probs else None

    return Predictions(y_true=y_true, y_pred=y_pred, logits=out_logits, probs=out_probs)


@torch.inference_mode()
def predict_tensor(model: torch.nn.Module, x: torch.Tensor, *, device: torch.device):
    """Predict a single tensor."""
    model.eval()
    model.to(device)
    x = x.to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    return pred.cpu(), probs.cpu()


def evaluate_split(predictions: Predictions, class_names: list[str]) -> None:

    y_true = predictions.y_true.numpy()
    y_pred = predictions.y_pred.numpy()
    acc = (y_true == y_pred).mean()
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Accuracy per class: diagonal / row sum (true positives / all true samples for that class)
    acc_per_class = {}
    for i, class_name in enumerate(class_names):
        class_total = cm[i, :].sum()
        if class_total > 0:
            acc_per_class[class_name] = cm[i, i] / class_total
        else:
            acc_per_class[class_name] = 0.0

    return {"acc": acc, "cm": cm, "report": report, "acc_per_class": acc_per_class}


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str]):

    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()

    return fig


def plot_random_misclassified_cases(
    predictions: Predictions,
    data_loader: DataLoader,
    class_names: list[str],
    n_samples: int = 5,
    mean: list[float] | None = None,
    std: list[float] | None = None,
):
    """
    Plots random misclassified examples.

    Args:
        predictions: The Predictions object returned by predict_loader.
        data_loader: The data loader used to generate predictions.
                     CRITICAL: Must be the SAME loader (or one with shuffle=False and same order).
                     If the loader was shuffled during prediction, indices cannot be mapped back to the dataset easily.
        class_names: List of class names.
        n_samples: Number of samples to plot.
        mean: Normalization mean (for denormalization).
        std: Normalization std (for denormalization).
    """
    if std is None:
        std = [0.229, 0.224, 0.225]
    if mean is None:
        mean = [0.485, 0.456, 0.406]

    y_true = predictions.y_true
    y_pred = predictions.y_pred

    # ensure tensors are on CPU
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu()

    if y_true is None:
        logger.warning("y_true is None in predictions, cannot compute misclassified cases.")
        return plt.figure()

    misclassified_mask = y_true != y_pred
    misclassified_idx = torch.where(misclassified_mask)[0]

    if len(misclassified_idx) == 0:
        logger.info("No misclassified cases found!")
        return plt.figure()

    # select random samples:
    n_plot = min(n_samples, len(misclassified_idx))
    random_idx = torch.randperm(len(misclassified_idx))
    selected_idx = misclassified_idx[random_idx]

    # Prepare figure
    cols = 3
    rows = (n_plot + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if n_plot == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    dataset = data_loader.dataset

    for ax, idx in zip(axes, selected_idx, strict=False):
        idx = idx.item()

        # Access image and label from dataset
        # Note: This assumes the dataset is indexable and the predicted order matches dataset order
        img, label = dataset[idx]

        # Denormalize image for plotting if it's a tensor
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
            img = (img * std) + mean
            img = np.clip(img, 0, 1)

        true_label_idx = y_true[idx].item()
        pred_label_idx = y_pred[idx].item()

        true_name = class_names[true_label_idx] if true_label_idx < len(class_names) else str(true_label_idx)
        pred_name = class_names[pred_label_idx] if pred_label_idx < len(class_names) else str(pred_label_idx)

        # Get probability if available
        prob_str = ""
        if predictions.probs is not None:
            prob = predictions.probs[idx, pred_label_idx].item()
            prob_str = f"({prob:.2f})"

        ax.imshow(img)
        ax.set_title(f"True: {true_name}\nPred: {pred_name}{prob_str}", color="red", fontsize=10)
        ax.axis("off")

    # Turn off remaining empty axes
    for i in range(n_plot, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    return fig


def log_eval_to_mlflow(metrics: dict, fig, split_name: str) -> None:

    mlflow.log_metric(f"{split_name} / acc ", metrics["acc"])
    mlflow.log_figure(fig, f"{split_name}_confusion_matrix / confusion_matrix.png")

    # F1 for each class
    for cls in [k for k in metrics["report"].keys() if k not in ("accuracy", "macro avg", "weighted avg")]:
        mlflow.log_metric(f"{split_name}/f1/{cls}", float(metrics["report"][cls]["f1-score"]))

    # Accuracy per class
    for cls, acc_value in metrics["acc_per_class"].items():
        mlflow.log_metric(f"{split_name}/acc/{cls}", float(acc_value))
