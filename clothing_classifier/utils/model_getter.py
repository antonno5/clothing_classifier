import torch
from transformers import (
    AutoImageProcessor,
    ResNetConfig,
    ResNetForImageClassification,
)


MODEL_SOURCE = "microsoft/resnet-50"


def get_model() -> torch.nn.Module:
    config = ResNetConfig(
        num_labels=10,
        depths=[2, 2, 2, 2],
        hidden_sizes=[32, 64, 64, 32],
        layer_type="basic",
        num_channels=3,
        image_size=224,
    )
    return ResNetForImageClassification(config)


def get_processor() -> AutoImageProcessor:
    processor = AutoImageProcessor.from_pretrained(MODEL_SOURCE, use_fast=True)
    return processor
