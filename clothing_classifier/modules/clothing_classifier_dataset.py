import numpy as np
import pandas as pd
import PIL.Image as Image
import torchvision.transforms as transforms
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

from utils.label_handler import get_label_list


class ClothingClassificationDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        processor: AutoImageProcessor,
        transform: transforms.Compose = None,
    ) -> None:
        self._dataset = dataset
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self._dataset)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        processor: AutoImageProcessor,
        transform: transforms.Compose = None,
    ) -> "ClothingClassificationDataset":
        result = cls(dataset, processor, transform)
        result.df = pd.DataFrame()
        batch_size = 1000
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            if isinstance(batch, dict):
                batch_df = pd.DataFrame.from_dict(batch)
            else:
                batch_df = batch.to_pandas()
            result.df = pd.concat([result.df, batch_df])
        result.labels = get_label_list()
        result.label2id = {label: i for i, label in enumerate(result.labels)}
        result.id2label = {i: label for i, label in enumerate(result.labels)}
        return result

    @property
    def labels_list(self):
        return self.labels

    @property
    def dataset(self):
        return self._dataset

    def __getitem__(self, idx):
        img = np.array(self.df.iloc[idx]["pixels"])
        label = int(self.df.iloc[idx]["label"])

        image = Image.fromarray(img, mode="L")

        image = self.transform(image)

        return {
            "pixel_values": image,
            "labels": label,
        }

    @classmethod
    def from_data_path(
        cls,
        csv_file: str,
        processor: AutoImageProcessor,
        transform: transforms.Compose = None,
    ) -> "ClothingClassificationDataset":
        df = pd.read_csv(csv_file)

        pixel_columns = df.columns[1:]
        new_df = pd.DataFrame(
            {
                "label": df["label"].astype(int),
                "pixels": df[pixel_columns].astype(float).values.tolist(),
            }
        )
        result = ClothingClassificationDataset.from_dataset(
            HFDataset.from_pandas(new_df), processor, transform
        )
        result.df = df
        return result
