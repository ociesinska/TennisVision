import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    convnext_tiny,
    efficientnet_b0,
    mobilenet_v3_large,
    resnet18,
    resnet50,
)


def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_head(model):
    for name, p in model.named_parameters():
        if any(k in name for k in ["fc", "classifier", "heads.head"]):
            p.requires_grad = True


def unfreeze_resnet_layer4(model) -> None:
    # works only for resnet models
    if hasattr(model, "layer4"):
        for p in model.layer4.parameters():
            p.requires_grad = True


def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    name = model_name.lower()

    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, weights

    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, weights

    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model, weights

    if name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model, weights

    if name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model, weights

    # TODO: model - touch / no touch -> ? how to build this

    raise ValueError(f"Unknown model_name: {model_name}")
