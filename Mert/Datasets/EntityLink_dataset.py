from dataclasses import dataclass
from typing import Tuple
from torch.utils.data import Dataset
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 2300000000
import json
import os
from transformers import FlavaProcessor, BertTokenizerFast
from .DataLoaderX import DataLoaderX
import torch
import numpy as np
import random


@dataclass
class EntityLinkDatasetConfig:
    max_length: int = 64
    abs_max_length: int = 128
    train_MEL_path: str = '/home/zero_lag/Document/srtp/Multimodality-Link/MELdataset/KVQA/train_v2.json'
    val_MEL_path: str = '/home/zero_lag/Document/srtp/Multimodality-Link/MELdataset/KVQA/dev_v2.json'
    KG_path: str = '/home/zero_lag/Document/srtp/Multimodality-Link/MELdataset/KVQA/KG.json'
    img_path: str = '/home/zero_lag/Document/srtp/Multimodality-Link/MELdataset/KVQA/images'

    batch_size: int = 8
    num_workers: int = 8
    shuffle_seed: int = 10086

def abs_dict_to_str(x: dict):
    if type(x)==type('str'):
        return x
    else:
        return str(x).replace('{', '').replace('}', '').replace('\'', '')

class EntityLinkDataset(Dataset):
    def __init__(self, MEL_path, KG_path, img_path) -> None:
        super().__init__()
        self.data = self.load_data(MEL_path, KG_path, img_path)

    def load_data(self, MEL_path: str, KG_path: str, img_path: str) -> list:
        with open(MEL_path) as f:
            dataset = json.load(f)
        with open(KG_path) as f:
            kg = json.load(f)
        Data = []
        for key in dataset.keys():
            sentence = dataset[key]['sentence']
            mention = dataset[key]['mentions']
            img = os.path.join(img_path, dataset[key]['imgPath'].split('/')[-1])
            positive_abstract = abs_dict_to_str(kg[dataset[key]['answer']])
            negative_abstract = [
                abs_dict_to_str(kg[cand_id]) for cand_id in dataset[key]['candidates']
            ]
            Data.append({
                'sentence': sentence,
                'mention': mention,
                'image': img,
                'positive_abstract': positive_abstract,
                'negative_abstract': negative_abstract
            })
        return Data



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class EntityLinkDatasetCollateFn:
    def __init__(self, multi_processor: FlavaProcessor, bert_processor: BertTokenizerFast,
                 config: EntityLinkDatasetConfig) -> None:
        self.multi_processor = multi_processor
        self.bert_processor = bert_processor
        self.config = config

    def __call__(self, batch_samples):
        batch_sentence, batch_mention, batch_img, batch_positive, batch_negative_inputs = [],[],[],[],[]

        for sample in batch_samples:
            batch_sentence.append(sample['sentence'])
            batch_mention.append(sample['mention'])
            batch_img.append(Image.open(sample['image']).convert('RGB'))
            batch_positive.append(sample['positive_abstract'])
            batch_negative_inputs.append(
                self.bert_processor(sample['negative_abstract'],
                                    return_tensors="pt",
                                    padding="max_length",
                                    max_length=self.config.abs_max_length,
                                    truncation=True))
        multi_input = self.multi_processor(text=batch_sentence,
                                           images=batch_img,
                                           return_tensors="pt",
                                           padding="max_length",
                                           max_length=self.config.max_length,
                                           truncation=True)
        batch_mention_token_pos = []
        for s, m in enumerate(batch_mention):
            mention_tokens = self.multi_processor(text=m, add_special_tokens=False).tokens()
            try:
                start = multi_input.tokens(s).index(mention_tokens[0])
                end = multi_input.tokens(s).index(mention_tokens[-1])
            except ValueError:
                start,end=0,0
            batch_mention_token_pos.append((start, end))

        batch_positive_inputs = self.bert_processor(batch_positive,
                                                    return_tensors="pt",
                                                    padding="max_length",
                                                    max_length=self.config.abs_max_length,
                                                    truncation=True)

        return multi_input, batch_mention_token_pos, batch_positive_inputs, batch_negative_inputs


def getEntityLinkDataLoader(config: EntityLinkDatasetConfig = EntityLinkDatasetConfig()) -> Tuple[
        DataLoaderX, DataLoaderX]:
    train_dataset = EntityLinkDataset(config.train_MEL_path, config.KG_path, config.img_path)
    val_dataset = EntityLinkDataset(config.val_MEL_path, config.KG_path, config.img_path)
    multi_processor = FlavaProcessor.from_pretrained('facebook/flava-full')
    bert_processor = BertTokenizerFast.from_pretrained('bert-base-cased')
    collateFn = EntityLinkDatasetCollateFn(multi_processor, bert_processor, config)
    train_dataloader = DataLoaderX(train_dataset,
                                   batch_size=config.batch_size,
                                   collate_fn=collateFn,
                                   shuffle=True,
                                   num_workers=config.num_workers,
                                   pin_memory=True)
    val_dataloader = DataLoaderX(val_dataset,
                                 batch_size=config.batch_size,
                                 collate_fn=collateFn,
                                 shuffle=True,
                                 num_workers=config.num_workers,
                                 pin_memory=True)

    torch.manual_seed(config.shuffle_seed)
    torch.cuda.manual_seed_all(config.shuffle_seed)
    np.random.seed(config.shuffle_seed)
    random.seed(config.shuffle_seed)
    torch.backends.cudnn.deterministic = True
    return train_dataloader, val_dataloader
