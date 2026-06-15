from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tennisvision.tasks.detection.types import DetectionResult


def viz_detected_boxes(
    detection_result: DetectionResult,
    save_path: str | Path | None = None,
):

    if not detection_result.image_path:
        raise ValueError("DetectionResult.image_path is required to visualize detection result.")

    image = Image.open(detection_result.image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for detection in detection_result.detections:
        x1, y1, x2, y2 = detection.xyxy
        label = f"{detection.label} {detection.confidence:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
        text_x = x1
        text_y = max(0, y1 - 12)
        draw.text((text_x, text_y), label, fill=(255, 0, 0), font=font)

    if save_path is not None:
        output_path = Path(save_path)
        if not output_path.suffix:
            image_name = Path(detection_result.image_path).stem
            output_path = output_path / f"{image_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    return image
