from pathlib import Path
from shutil import copytree, rmtree

DATASET_ROOT = Path("data/detection")
EXPORT_ROOT = Path("cvat_export")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val")


def main() -> None:
    images_source = DATASET_ROOT / "images"
    labels_source = DATASET_ROOT / "labels"

    if not images_source.exists():
        raise FileNotFoundError(f"Folder with images not found: {images_source}")

    if not labels_source.exists():
        raise FileNotFoundError(f"Folder with labels not found: {labels_source}")

    if EXPORT_ROOT.exists():
        rmtree(EXPORT_ROOT)

    EXPORT_ROOT.mkdir(parents=True)
    copytree(images_source, EXPORT_ROOT / "images")
    copytree(labels_source, EXPORT_ROOT / "labels")

    for split in SPLITS:
        image_dir = EXPORT_ROOT / "images" / split
        label_dir = EXPORT_ROOT / "labels" / split

        image_paths = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        missing_labels = [image_path.name for image_path in image_paths if not (label_dir / f"{image_path.stem}.txt").exists()]

        if missing_labels:
            examples = "\n".join(missing_labels[:10])
            raise FileNotFoundError(f"Missing labels for {len(missing_labels)} images in split: {split}.\nExamples:\n{examples}")

        lines = [f"images/{split}/{image_path.name}" for image_path in image_paths]
        (EXPORT_ROOT / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        print(f"{split}: {len(lines)} images")

    data_yaml = """path: ./

train: train.txt
val: val.txt

names:
  0: player
"""

    (EXPORT_ROOT / "data.yaml").write_text(data_yaml, encoding="utf-8")
    print(f"Ready CVAT folder: {EXPORT_ROOT}")


if __name__ == "__main__":
    main()
