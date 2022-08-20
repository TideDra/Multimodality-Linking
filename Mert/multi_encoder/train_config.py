from dataclasses import dataclass, field
from typing import Tuple

from datasets.DataLoaderX import DataLoaderX
from datasets.flickr_dataset import getFlickrDataLoader
from multi_encoder.config import MultiEncoderConfig
from multi_encoder.model import MultiEncoder, MultiEncoderV2_2


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

    encoder: MultiEncoderV2_2 = field(
        default_factory=lambda: MultiEncoderV2_2(config=MultiEncoderConfig(d_text=64, d_vision=197))
    )
    dataloader: Tuple[DataLoaderX, DataLoaderX] = field(default_factory=getFlickrDataLoader)
