from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from captum.attr import LayerAttribution, LayerGradCam
from PIL import Image


def get_last_conv_layer(model: nn.Module) -> nn.Module:
    #TODO
    raise NotImplementedError

def preprocess_PIL(img: Image.Image, preprocess) -> torch.Tensor:
    return preprocess(img.convert("RGB")).unsqueeze(0)

def gradcam_heatmap(
        model: nn.Module,
        x: torch.Tensor,
        target: int,
        conv_layer: nn.Module,
        device: torch.device
        ) -> np.ndarray:
        model.eval()
        model.to(device)
        x = x.to(device)
        x.requires_grad_(True)

        cam = LayerGradCam(model, conv_layer)
        attr = cam.attribute(x, target) # here captum does forward step + backward, calculates grad-cam for class target
        # attr is [1, 1, h, w] or [1, h, w]
        heat = LayerAttribution.interpolate(attr, x.shape[-2:]) # adjusting the heatmap to the input size 
        heat = heat.squeeze().detach().cpu().numpy()
        heat = np.maximum(heat, 0)
        heat = heat / (heat.max() + 1e-8)

        return heat

def pick_cam_layer(model: nn.Module, model_name: str) -> nn.Module:
     #TODO: change this methodology
    if "resnet" in model_name:
         return model.layer3[-1]
    if "mobilenet_v3" in model_name:
         return model.features[-1]
    raise ValueError(f"Unknown model type for model {model_name}")

def overlay_heatmap(img: Image.Image | np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_TURBO,
    is_rgb: bool = True):
    """
    heatmap: np.ndarray (H,W) float in [0,1]
    img: PIL.Image (RGB) OR np.ndarray (H,W,3) in RGB/BGR (see to_rgb)

    Returns: overlay image as np.ndarray (H, W, 3) in RGB by default.
    
    Parameter is_rgb: if False, the input image is in BGR and will be converted to RGB; if True, the input image is already in RGB and does not require conversion.
    """
    # image --> numpy (RGB)
    if isinstance(img, Image.Image):
        img_rgb = np.array(img.convert("RGB"))
    else:
        img_rgb = img
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError("img must be HxWx3")
        # if BGR:
        if is_rgb is False:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    H, W = img_rgb.shape[:2]

    # resize heatmap if needed
    if heatmap.shape != (H, W):
        heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        heatmap_resized = heatmap
    # heatmap -> uint8 (0,255)
    heatmap_u8 = np.clip(heatmap_resized * 255.0, 0, 255).astype(np.uint8)

    heatmap_color_bgr = cv2.applyColorMap(heatmap_u8, colormap)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_rgb, 1.0 - alpha, heatmap_color_rgb, alpha, 0.0)

    return overlay


def explainability_for_training(model, epoch, data_loader, device, explain_every=2, explain_sample=3):
    torch.manual_seed(42)

    if explain_sample > 5:
        explain_sample = 5
        raise Warning("explain_sample should not exceed 5. Setting explain_sample to 5.")

    if epoch % explain_every == 0:
        # taking a sample of images from each epoch
        rand_idx = torch.randperm(len(data_loader.dataset))[:explain_sample]
        
        for idx in rand_idx:
            img_tensor, _ = data_loader.dataset[idx]
            x = img_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred = model(x)
            top2_idx = pred.topk(k=2, dim=1).indices
            pred_idx1 = top2_idx[:, 0]
            pred_idx2 = top2_idx[:, 1]

            conv_layer = model.features[-2]  # TODO: this is adjusted for mobilenet, adjust to other model types
            heatmap_pred1 = gradcam_heatmap(model=model, x=x, target=pred_idx1, conv_layer=conv_layer, device=device)
            heatmap_pred2 = gradcam_heatmap(model=model, x=x, target=pred_idx2, conv_layer=conv_layer, device=device)

            # denormalize tensor back to image for overlay
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
            img_np = (img_np * std + mean) * 255.0
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            overlay_pred1 = overlay_heatmap(img_np, heatmap_pred1, alpha=0.4, is_rgb=True)
            overlay_pred2 = overlay_heatmap(img_np, heatmap_pred2, alpha=0.4, is_rgb=True)

        return overlay_pred1, overlay_pred2
    else:
        return None, None