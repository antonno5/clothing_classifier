import typing as tp

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, default_collate, random_split

import utils.label_handler as label_handler
from modules.clothing_classifier_dataset import ClothingClassificationDataset
from utils.data_handler import download_data, ensure_data_unpacked
from utils.model_getter import get_processor


class ResNetDataModule(pl.LightningDataModule):
    _processor = get_processor()

    _train_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    _val_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    def __init__(self, config: dict[str, any]) -> None:
        super().__init__()
        self._data_csv = config["data"]["data_path"]
        self._config = config
        self.batch_size = config["training"]["batch_size"]
        self.num_workers = config["training"]["num_workers"]
        self.data_split_ratio = config["training"]["train_val_ratio"]
        self._seed = config["training"]["seed"]
        self._generator = torch.Generator().manual_seed(config["training"]["seed"])

    # @staticmethod
    # def collate_fn(examples: dict[str, any]) -> dict[str, any]:
    #     print(examples)
    #     pixel_values = torch.stack(
    #         [example["pixel_values"] for example in examples], dim=0
    #     )
    #     labels = torch.stack([example["labels"] for example in examples])
    #     return {"pixel_values": pixel_values, "labels": labels}

    def train_test_split(
        self, dataset: torch.utils.data.Dataset
    ) -> tuple[ClothingClassificationDataset, ClothingClassificationDataset]:
        train_ratio = self.data_split_ratio

        train_dataset, test_dataset = random_split(
            dataset, [train_ratio, 1.0 - train_ratio], generator=self._generator
        )
        custom_train_dataset = ClothingClassificationDataset.from_dataset(
            train_dataset,
            processor=get_processor(),
            transform=ResNetDataModule._train_transforms,
        )
        custom_test_dataset = ClothingClassificationDataset.from_dataset(
            test_dataset,
            processor=get_processor(),
            transform=ResNetDataModule._val_transforms,
        )
        return custom_train_dataset, custom_test_dataset

    def train_val_split(
        self, dataset: torch.utils.data.Dataset
    ) -> tuple[ClothingClassificationDataset, ClothingClassificationDataset]:
        train_ratio = self.data_split_ratio

        train_dataset, val_dataset = random_split(
            dataset, [train_ratio, 1.0 - train_ratio], generator=self._generator
        )
        custom_train_dataset = ClothingClassificationDataset.from_dataset(
            train_dataset,
            processor=get_processor(),
            transform=ResNetDataModule._train_transforms,
        )
        custom_val_dataset = ClothingClassificationDataset.from_dataset(
            val_dataset,
            processor=get_processor(),
            transform=ResNetDataModule._val_transforms,
        )
        return custom_train_dataset, custom_val_dataset

    def prepare_data(self):
        try:
            ensure_data_unpacked(self._data_csv)
        except Exception:
            print("Failed getting data from dvc, downloading...")
            download_data()

    def setup(self, stage: tp.Optional[str] = None):
        self._num_labels = self._config["model"]["num_labels"]
        self.dataset = ClothingClassificationDataset.from_data_path(
            self._data_csv,
            processor=get_processor(),
            transform=ResNetDataModule._val_transforms,
        )
        self.labels = sorted(list(set(self.dataset.labels)))
        self.id2label = self.dataset.id2label
        label_handler.write_labels_metainfo(
            self._config["data"]["id2labels_meta"], self.id2label
        )
        self.label2id = self.dataset.label2id
        label_handler.write_labels_metainfo(
            self._config["data"]["labels2id_meta"], self.label2id
        )

        self.custom_train_dataset, self.custom_test_dataset = self.train_test_split(
            self.dataset.dataset
        )
        self.custom_train_dataset, self.custom_val_dataset = self.train_val_split(
            self.custom_train_dataset.dataset
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.custom_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=default_collate,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.custom_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=default_collate,
            persistent_workers=True,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.custom_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=default_collate,
        )
