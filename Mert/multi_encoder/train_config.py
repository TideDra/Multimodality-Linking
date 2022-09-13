from dataclasses import dataclass, field
from typing import Tuple

from Datasets.DataLoaderX import DataLoaderX
from Datasets.flickr_dataset import getFlickrDataLoader
from multi_encoder.model import MultiEncoder


@dataclass
class MultiEncoderTrainConfig:
    epochs = 1
    learning_rate = 5e-5
    ckpt_name = "multi-encoder-flickr"
    ckpt_path = "/home/model_ckpt"
    state_path = "/home/model_state"
    save_state_interval = 100
    max_ckpt_num = 5
    load_ckpt = True

    encoder: MultiEncoder = field(default_factory=lambda: MultiEncoder())
    dataloader: Tuple[DataLoaderX, DataLoaderX] = field(default_factory=getFlickrDataLoader)
