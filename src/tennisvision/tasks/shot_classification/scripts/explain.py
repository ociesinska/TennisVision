import argparse
import glob
import json
import logging
import os
import time

import mlflow
from mlflow import MlflowClient
from PIL import Image

from tennisvision.core.data import build_preprocess
from tennisvision.core.explainability import gradcam_heatmap, overlay_heatmap, pick_cam_layer, preprocess_PIL
from tennisvision.core.mlflow_utils import load_model_from_mlflow, setup_mlflow
from tennisvision.core.utils import get_device, setup_logging


def main():
    # TODO: add max pics for explainability 
    parser = argparse.ArgumentParser(description="Explainability for TennisVision")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model_uri", type=str, help="MLflow model URI (e.g. models:/TennisVision@champion)")
    model_group.add_argument("--run_id", type=str, help="MLflow run ID (e.g. abc123456789)")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image file")
    group.add_argument("--input-dir", type=str, help="Path to a directory with images")

    parser.add_argument("--model_name", type=str, default="resnet18", help="Model architecture name (e.g. resnet18, mobilenet_v3_large)")

    args = parser.parse_args()

    setup_logging(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Explainability started.")
    device = get_device()
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
    setup_mlflow(experiment_name="Explainability", tracking_uri=mlflow_tracking_uri, set_experiment=True)

    with mlflow.start_run(run_name =  f"explain-{time.strftime('Y%m%d_%H%M%S')}"):

        if args.model_uri:
            model, model_uri = load_model_from_mlflow(model_uri=args.model_uri, device=device)
        elif args.run_id:
            model, model_uri = load_model_from_mlflow(run_id=args.run_id, device=device)

        model.eval()
        mlflow.log_param("source_model_uri", model_uri)
        
        if args.image:
            image_list = [args.image]
        elif args.input_dir:
            patterns = ["*.jpeg", "*.jpg", "*.png"]
            image_list = []
            for pattern in patterns:
                image_list.extend(glob.glob(os.path.join(args.input_dir, pattern)))
            image_list.sort()
        else:
            raise ValueError("Specify --image or --input-dir")

        if args.run_id:
            client = MlflowClient(tracking_uri=mlflow_tracking_uri)
            idx_to_class_path = client.download_artifacts(args.run_id, "labels/idx_to_class.json")
            with open(idx_to_class_path) as x:
                idx_to_class =json.load(x)
                idx_to_class = {int(k): v for k, v in idx_to_class.items()}
        else: # TODO: adjust to avoid hardcoding
            idx_to_class = {0: "backhand", 1: "forehand", 2: "ready_position", 3: "serve"}
        
        preprocess = build_preprocess()
        model_name = args.model_name
        conv_layer = pick_cam_layer(model=model, model_name=model_name)

        for i, img in enumerate(image_list):
            image = Image.open(img)
            x = preprocess_PIL(image, preprocess)
            x = x.to(device)
            pred = model(x) # [B, C]
            top2_idx = pred.topk(k=2, dim=1).indices # [B, 2]
            pred_idx1 = top2_idx[:, 0]
            pred_idx2 = top2_idx[:, 1]
            pred_class_1 = idx_to_class[pred_idx1.item()]
            pred_class_2 = idx_to_class[pred_idx2.item()]

            heatmap_pred1 = gradcam_heatmap(model=model, x=x, target=pred_idx1, conv_layer=conv_layer, device=device)
            heatmap_pred2 = gradcam_heatmap(model=model, x=x, target=pred_idx2, conv_layer=conv_layer, device=device)

            overlay_pred1 = overlay_heatmap(image, heatmap_pred1, alpha=0.4, is_rgb=True)
            overlay_pred2 = overlay_heatmap(image, heatmap_pred2, alpha=0.4, is_rgb=True)

            mlflow.log_image(overlay_pred1, f"Grad-CAM/heatmap_pred1_{i}_{pred_class_1}.png")
            mlflow.log_image(overlay_pred2, f"Grad-CAM/heatmap_pred2_{i}_{pred_class_2}.png")

if __name__ == "__main__":
    main()