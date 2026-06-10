from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tennisvision.tasks.detection.inference import DetectionInferenceConfig, load_detector, predict_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detection inference on one image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DetectionInferenceConfig.model_path)
    parser.add_argument("--backend", type=str, default="ultralytics", choices=["ultralytics", "torchvision"])
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--visualize",
        type=lambda value: str(value).lower() in {"1", "true", "yes", "y"},
        default=False,
    )
    args = parser.parse_args()

    cfg = DetectionInferenceConfig(
        backend=args.backend,
        model_path=args.model_path,
        imgsz=args.imgsz,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
        visualize=args.visualize,
    )

    model = load_detector(cfg)
    result = predict_image(model, args.image, cfg)

    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
