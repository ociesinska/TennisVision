from pathlib import Path
from shutil import copy2, make_archive, rmtree

DATASET_ROOT = Path("data/detection")
EXPORT_ROOT = Path("cvat_export")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("test",)
CLASS_NAMES = ("player",)


def prepare_split(split: str) -> None:
    images_source = DATASET_ROOT / "images" / split
    labels_source = DATASET_ROOT / "labels" / split

    if not images_source.exists():
        raise FileNotFoundError(f"Folder with images not found: {images_source}")

    if not labels_source.exists():
        raise FileNotFoundError(f"Folder with labels not found: {labels_source}")

    split_export_root = EXPORT_ROOT / split
    if split_export_root.exists():
        rmtree(split_export_root)

    image_dir = split_export_root / "images" / split
    label_dir = split_export_root / "labels" / split
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image_paths = sorted(path for path in images_source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    missing_labels = [image_path.name for image_path in image_paths if not (labels_source / f"{image_path.stem}.txt").exists()]

    if missing_labels:
        examples = "\n".join(missing_labels[:10])
        raise FileNotFoundError(f"Missing labels for {len(missing_labels)} images in split: {split}.\nExamples:\n{examples}")

    for image_path in image_paths:
        label_path = labels_source / f"{image_path.stem}.txt"
        copy2(image_path, image_dir / image_path.name)
        copy2(label_path, label_dir / label_path.name)

    (split_export_root / "data.yaml").write_text(
        "path: .\n"
        f"train: images/{split}\n"
        f"val: images/{split}\n"
        f"test: images/{split}\n\n"
        "names:\n" + "".join(f"  {class_id}: {class_name}\n" for class_id, class_name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )

    archive_path = make_archive(str(EXPORT_ROOT / split), "zip", root_dir=split_export_root)

    print(f"{split}: {len(image_paths)} images")
    print(f"Ready CVAT folder: {split_export_root}")
    print(f"Ready CVAT zip: {archive_path}")


def main() -> None:
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        prepare_split(split)


if __name__ == "__main__":
    main()
