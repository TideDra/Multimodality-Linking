from typing import Tuple
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from transformers import FlavaProcessor, BertTokenizerFast
from .DataLoaderX import DataLoaderX
import torch
import numpy as np
import random

class EntityLinkDatasetConfig:
    max_length = 32
    abs_max_length = 128
    train_MEL_path='/home/zero_lag/Document/srtp/Multimodality-Link/MELdataset/RichpediaMEL/Richpedia-MELv2_train.json'
    val_MEL_path='/home/zero_lag/Document/srtp/Multimodality-Link/MELdataset/RichpediaMEL/Richpedia-MELv2_val.json'
    KG_path='/home/zero_lag/Document/srtp/Multimodality-Link/MELdataset/RichpediaMEL/Richpedia-KGv2.json'
    img_path=None

    batch_size=8
    num_workers=8
    shuffle_seed=10086
class EntityLinkDataset(Dataset):
    def __init__(self, MEL_path,KG_path,img_path) -> None:
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
            img = os.path.join(img_path, f'{key}.jpg')
            positive_abstract = self.__dict_to_str(kg[key])
            negative_abstract = [
                self.__dict_to_str(kg[cand_id])
                for cand_id in dataset[key]['candidates']
            ]
            Data.append({
                'sentence': sentence,
                'mention': mention,
                'image': img,
                'positive_abstract': positive_abstract,
                'negative_abstract': negative_abstract
            })
        return Data

    def __dict_to_str(self, x: dict):
        return str(x).replace('{', '').replace('}', '').replace('\'', '')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class EntityLinkDatasetCollateFn:
    def __init__(self, multi_processor:FlavaProcessor,bert_processor:BertTokenizerFast,config:EntityLinkDatasetConfig) -> None:
        self.multi_processor=multi_processor
        self.bert_processor=bert_processor
        self.config=config
    
    def __call__(self, batch_samples):
        batch_sentence, batch_mention, batch_img, batch_positive, batch_negative_inputs = [[]]*5

        for sample in batch_samples:
            batch_sentence.append(sample['sentence'])
            batch_mention.append(sample['mention'])
            batch_img.append(Image.open(sample['image']))
            batch_positive.append(sample['positive_abstract'])
            batch_negative_inputs.append(
                self.bert_processor(sample['negative_abstract'],
                               return_tensors="pt",
                               padding="max_length",
                               max_length=self.config.abs_max_length,
                               truncation=True))
        multi_input = self.multi_processor(
            text=batch_sentence,
            images=batch_img,
            return_tensors="pt",
            padding="max_length",
            max_length=self.config.max_length,
            truncation=True)
        batch_mention_token_pos = []
        for s, m in enumerate(batch_mention):
            mention_pos = batch_sentence[s].split(' ').index(m)
            batch_mention_token_pos.append(multi_input.word_to_tokens(s, mention_pos))

        batch_positive_inputs=self.bert_processor(batch_positive)

        return multi_input,batch_mention_token_pos,batch_positive_inputs,batch_negative_inputs

    
def getEntityLinkDataLoader(config:EntityLinkDatasetConfig=EntityLinkDatasetConfig())->Tuple(DataLoaderX,DataLoaderX):
    train_dataset=EntityLinkDataset(config.train_MEL_path,config.KG_path,config.img_path)
    val_dataset=EntityLinkDataset(config.val_MEL_path,config.KG_path,config.img_path)
    multi_processor = FlavaProcessor.from_pretrained('facebook/flava-full')
    bert_processor = BertTokenizerFast.from_pretrained('bert-base-cased')
    collateFn=EntityLinkDatasetCollateFn(multi_processor,bert_processor)
    train_dataloader=DataLoaderX(train_dataset,batch_size=config.batch_size,collate_fn=collateFn,shuffle=True,num_workers=config.num_workers,pin_memory=True)
    val_dataloader=DataLoaderX(val_dataset,batch_size=config.batch_size,collate_fn=collateFn,shuffle=True,num_workers=config.num_workers,pin_memory=True)

    torch.manual_seed(config.shuffle_seed)
    torch.cuda.manual_seed_all(config.shuffle_seed)
    np.random.seed(config.shuffle_seed)
    random.seed(config.shuffle_seed)
    torch.backends.cudnn.deterministic = True
    return train_dataloader,val_dataloader
