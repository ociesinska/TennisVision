# TennisVision 🎾

Computer Vision pipeline for tennis action classification using PyTorch and transfer learning.

## Project Overview

TennisVision is an ML system that classifies tennis player actions into 4 categories:
- **Backhand** - Backhand stroke
- **Forehand** - Forehand stroke  
- **Ready Position** - Player in ready stance
- **Serve** - Service motion

Uses pretrained models (ResNet, EfficientNet, MobileNet, ConvNeXt) with:
- Head-only training (frozen backbone)
- Optional fine-tuning (layer4 unfrozen)
- MLflow experiment tracking
- Optuna hyperparameter optimization

## Installation

```bash
# Clone repo
git clone https://github.com/ociesinska/TennisVision.git
cd TennisVision

# Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .