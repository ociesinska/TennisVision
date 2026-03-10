import io
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S", force=True)

def get_default_out_dir():
    timestamp = time.strftime("Y%m%d_%H%M%S")
    return f"artifacts/explain/{timestamp}"

def get_model_name() -> str:
    # TODO
    return

def rgb_ndarray_to_png_bytes(arr: np.ndarray) -> bytes:
    # arr: (H, W, 3) in RGB
    if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def concat_rgb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a,b: RGB uint8 (H,W,3)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if b.dtype != np.uint8:
        b = np.clip(b, 0, 255).astype(np.uint8)

    if a.shape != b.shape:
        b_img = Image.fromarray(b, mode="RGB").resize((a.shape[1], a.shape[0]))
        b = np.array(b_img, dtype=np.uint8)

    return np.concatenate([a, b], axis=0)

