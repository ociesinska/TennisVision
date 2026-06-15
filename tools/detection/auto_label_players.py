from pathlib import Path

from ultralytics import YOLO

DATASET_ROOT = Path("data/detection")
MODEL_PATH = "yolo11m.pt"
SPLITS = ("train", "val")


def auto_label_split(model: YOLO, split: str, confidence: float = 0.15, image_size: int = 1280) -> None:
    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    label_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(image_dir),
        classes=[0],
        conf=confidence,
        imgsz=image_size,
        stream=True,
        verbose=False,
    )

    processed = 0

    for result in results:
        image_path = Path(result.path)
        label_path = label_dir / f"{image_path.stem}.txt"
        lines: list[str] = []

        if result.boxes is not None:
            for x_center, y_center, width, height in result.boxes.xywhn.cpu().tolist():
                lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        label_path.write_text("\n".join(lines), encoding="utf-8")
        processed += 1

    print(f"{split}: generated labels for {processed} images")


def main() -> None:
    model = YOLO(MODEL_PATH)

    for split in SPLITS:
        auto_label_split(model=model, split=split)


if __name__ == "__main__":
    main()
