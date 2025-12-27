import pytorch_lightning as pl
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor

MODEL_SOURCE = "microsoft/resnet-50"

def get_model(datamodule: pl.LightningDataModule) -> torch.nn.Module:
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_SOURCE,
        num_labels=len(datamodule.id2label),
        id2label=datamodule.id2label,
        label2id=datamodule.label2id,
    )
    return model


def get_processor() -> AutoImageProcessor:
    processor = AutoImageProcessor.from_pretrained(MODEL_SOURCE)
    return processor
