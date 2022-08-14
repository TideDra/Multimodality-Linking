from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
import torch.utils.data
from .DataLoaderX import DataLoaderX
from PIL import Image
from torch.utils.data import Dataset
from transformers import FlavaProcessor


class FlickrDataset(Dataset):
    '''Dataset for Flickr30k'''
    def __init__(self, folder_path_str: str = None, cached: bool = True):
        self.data = self.load_data(folder_path_str)

    def load_data(self, folder_path_str: str = None):
        folder_path = Path(folder_path_str)
        annotations = pd.read_table(
            folder_path / 'results_20130124.token', sep='\t', header=None, names=['image', 'caption']
        )
        data = [{
            "image": folder_path / "flickr30k-images" / row["image"][:-2],
            "caption": row["caption"]
        } for index, row in annotations.iterrows()]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class FlickrDatasetColloter:
    def __init__(self, processor: FlavaProcessor, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch_samples):
        batch_text = [sample["caption"] for sample in batch_samples]
        batch_images = [Image.open(sample["image"]).convert("RGB") for sample in batch_samples]
        batch_inputs = self.processor(
            text=batch_text,
            images=batch_images,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        return {"batch_inputs": batch_inputs}


@dataclass
class FlickrDatasetConfig:
    base_path = "/home/data/"
    data_path = base_path + "flickr 30k"
    batch_size = 20
    max_length = 64
    num_workers = 1
    processor = FlavaProcessor.from_pretrained("facebook/flava-full")
    split_seed = 114514


def splitDataset(whole_ds: Dataset, train_ratio: float, seed) -> Tuple[Dataset, Dataset]:
    len_ds = len(whole_ds)
    train_size = int(train_ratio * len_ds)
    val_size = len_ds - train_size
    return torch.utils.data.random_split(
        whole_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )


def getFlickrDataLoader(config: FlickrDatasetConfig = FlickrDatasetConfig()) -> Tuple[DataLoaderX, DataLoaderX]:
    '''Get flickr DataLoaders for train and validate'''
    whole_ds = FlickrDataset(config.data_path)
    train_ds, val_ds = splitDataset(whole_ds, 0.9, config.split_seed)
    train_dl = DataLoaderX(
        train_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=FlickrDatasetColloter(config.processor, config.max_length),
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_dl = DataLoaderX(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=FlickrDatasetColloter(config.processor, config.max_length),
        num_workers=config.num_workers,
        pin_memory=True
    )
    return train_dl, val_dl
