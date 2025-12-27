import typing as tp
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import utils.label_handler as label_handler
from modules.clothing_classifier_dataset import ClothingClassificationDataset
from utils.data_handler import download_data, ensure_data_unpacked
from utils.model_getter import get_processor


class ResNetDataModule(pl.LightningDataModule):
    _processor = get_processor()
    _image_mean, _image_std = _processor.feature_extractor.mean, _processor.feature_extractor.std
    _size = _processor.feature_extractor.size[0]
    
    _train_transforms = transforms.Compose(
        [
            transforms.Resize((_size, _size)),
            transforms.RandomRotation(90),
            transforms.RandomAdjustSharpness(2),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=_image_mean, std=_image_std),
        ]
    )

    _val_transforms = transforms.Compose(
        [
            transforms.Resize((_size, _size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_image_mean, std=_image_std),
        ]
    )

    def __init__(self, config: dict[str, any]) -> None:
        super().__init__()
        self._data_dir = config["data_loading"]["data_path"]
        self._config = config
        self.batch_size = config["training"]["batch_size"]
        self.num_workers = config["training"]["num_workers"]
        self.data_split_ratio = config["training"]["train_val_ratio"]
        self._seed = config["training"]["seed"]
        self._generator = torch.Generator().manual_seed(config["training"]["seed"])

    @staticmethod
    def train_transforms(examples: dict[str, any]) -> dict[str, any]:
        examples["pixel_values"] = [
            ResNetDataModule._train_transforms(image.convert("RGB"))
            for image in examples["images"]
        ]
        return examples

    @staticmethod
    def val_transforms(examples: dict[str, any]) -> dict[str, any]:
        examples["pixel_values"] = [
            ResNetDataModule._val_transforms(image.convert("RGB"))
            for image in examples["images"]
        ]
        return examples

    @staticmethod
    def collate_fn(examples: dict[str, any]) -> dict[str, any]:
        pixel_values = torch.stack(
            [example["pixel_values"] for example in examples], dim=0
        )
        labels = torch.stack([example["labels"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    def train_test_split(
        self, dataset: torch.utils.data.Dataset
    ) -> tuple[ClothingClassificationDataset, ClothingClassificationDataset]:
        total_size = len(dataset)
        train_ratio = self.data_split_ratio
        train_size = int(train_ratio * total_size)
        test_size = total_size - train_size

        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], generator=self._generator
        )
        custom_train_dataset = ClothingClassificationDataset.from_dataset(
            train_dataset,
            processor=get_processor(),
            transform=ResNetDataModule.train_transforms,
        )
        custom_test_dataset = ClothingClassificationDataset.from_dataset(
            test_dataset,
            processor=get_processor(),
            transform=ResNetDataModule.val_transforms,
        )
        return custom_train_dataset, custom_test_dataset

    def train_val_split(
        self, dataset: torch.utils.data.Dataset
    ) -> tuple[ClothingClassificationDataset, ClothingClassificationDataset]:
        total_size = len(dataset)
        train_ratio = self.data_split_ratio
        train_size = int(train_ratio * total_size)
        val_size = total_size - train_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=self._generator
        )
        custom_train_dataset = ClothingClassificationDataset.from_dataset(
            train_dataset,
            processor=get_processor(),
            transform=ResNetDataModule.train_transforms,
        )
        custom_val_dataset = ClothingClassificationDataset.from_dataset(
            val_dataset,
            processor=get_processor(),
            transform=ResNetDataModule.val_transforms,
        )
        return custom_train_dataset, custom_val_dataset

    def prepare_data(self):
        try:
            for data_path in self._data_dir:
                ensure_data_unpacked(data_path)
        except Exception:
            print("Failed getting data from dvc, downloading...")
            download_data()

    def setup(self):
        self._num_labels = self._config["model"]["num_labels"]
        self.dataset = ClothingClassificationDataset.from_data_path(
            self._data_dir,
            self._num_labels,
            processor=get_processor(),
            randomize=True,
            seed=self._seed,
            transform=ResNetDataModule.val_transforms,
        )
        self.labels = sorted(list(set(self.dataset.labels)))
        self.id2label = self.dataset.id2label
        label_handler.write_labels_metainfo(
            self._config["data_loading"]["id2labels_meta"], self.id2label
        )
        self.label2id = self.dataset.label2id
        label_handler.write_labels_metainfo(
            self._config["data_loading"]["labels2id_meta"], self.label2id
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
            collate_fn=ResNetDataModule.collate_fn,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.custom_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ResNetDataModule.collate_fn,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.custom_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ResNetDataModule.collate_fn,
        )
