import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from transformers import AutoImageProcessor


class ClothingClassificationDataset(Dataset):
    def __init__(
        self,
        csv_file,
        processor: AutoImageProcessor,
        transform: transforms.Compose = None,
    ):
        self.df = pd.read_csv(csv_file)
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx]["label"]
        pixel_values = self.df.iloc[idx, 1:].values.astype(np.uint8)
        pixel_values = pixel_values.reshape(28, 28)

        image = Image.fromarray(pixel_values, mode="L")

        if self.transform:
            image = self.transform(image)
        else:
            raise NotImplementedError("Transformer isn't sent")

        return {"pixel_values": image, "labels": torch.tensor(label)}

    @classmethod
    def from_data_path(
        cls,
        csv_file: str,
        processor: AutoImageProcessor,
        transform: transforms.Compose = None,
    ):
        return cls(csv_file, processor, transform)
