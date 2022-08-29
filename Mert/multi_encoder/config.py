from dataclasses import dataclass


@dataclass
class MultiEncoderConfig:
    num_layers: int = 6
    d_text: int = 64
    d_vision: int = 14**2 + 1
    d_bottleneck: int = 4
    hidden_size: int = 768
    nhead: int = 4
    dropout = 0.1
    batch_first = True


@dataclass
class TwitterDatasetTrainConfig:
    base_path = "/home/data/"
    train_file_path = base_path + "twitter2017/train.txt"
    train_img_path = base_path + "twitter2017_images"
    train_cache_path = base_path + "twitter2017/train_cache.pickle"
    val_file_path = base_path + "twitter2017/valid.txt"
    val_img_path = base_path + "twitter2017_images"
    val_cache_path = base_path + "twitter2017/dev_cache.pickle"