from collections import defaultdict
from pathlib import Path

DATASET_ROOT = Path("data/detection")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val")


def main() -> None:
    for split in SPLITS:
        image_dir = DATASET_ROOT / "images" / split
        label_dir = DATASET_ROOT / "labels" / split

        images = {path.stem: path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
        labels = {path.stem: path for path in label_dir.glob("*.txt")}

        grouped: dict[str, list[Path]] = defaultdict(list)
        for path in image_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                grouped[path.stem].append(path)

        duplicates = {stem: paths for stem, paths in grouped.items() if len(paths) > 1}

        print(f"\n{split}")
        print("Images without labels:")
        missing_labels = sorted(images.keys() - labels.keys())
        if missing_labels:
            for stem in missing_labels:
                print(f"  {images[stem]}")
        else:
            print("  None")

        print("Labels without images:")
        missing_images = sorted(labels.keys() - images.keys())
        if missing_images:
            for stem in missing_images:
                print(f"  {labels[stem]}")
        else:
            print("  None")

        print("Duplicate stems:")
        if duplicates:
            for stem, paths in duplicates.items():
                print(f"  {stem}:")
                for path in paths:
                    print(f"    {path.name}")
        else:
            print("  None")


if __name__ == "__main__":
    main()
