from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
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

    forward_link = False
    '''forward返回链表'''
    sole_flava = False
    '''只使用一个FlavaModel而非三合一。使用此选项时不能进行Flava多模态融合'''
    augment_text = True
    '''使用多层级表示增强text'''
    passes = ["mm", "bn"]
    '''融合顺序; Options: mm, bn, none'''


@dataclass
class TwitterDatasetTrainConfig:
    base_path = "/home/data/"
    train_file_path = base_path + "twitter2017/train.txt"
    train_img_path = base_path + "twitter2017_images"
    train_cache_path = base_path + "twitter2017/train_cache.pickle"
    val_file_path = base_path + "twitter2017/valid.txt"
    val_img_path = base_path + "twitter2017_images"
    val_cache_path = base_path + "twitter2017/dev_cache.pickle"
