from dataclasses import dataclass
from typing import Tuple, Union
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
    train_MEL_path: str = '/hy-tmp/KVQA/train_v2.json'
    val_MEL_path: str = '/hy-tmp/KVQA/dev_v2.json'
    test_MEL_path: str = '/hy-tmp/KVQA/dev_v2.json'
    KG_path: str = '/hy-tmp/KVQA/KGv2.json'
    img_path: str = '/hy-tmp/KVQA/KVQAimgs'

    batch_size: int = 16
    num_workers: int = 20
    shuffle_seed: int = 10086

def abs_dict_to_str(x: Union[dict, str]):
    if isinstance(x, str):
        return x
    if 'Birth' in x: del x['Birth']
    if 'Death' in x: del x['Death']
    if 'Religion' in x: del x['Religion']
    if 'Spouse' in x: del x['Spouse']
    if 'Alma mater' in x: del x['Alma mater']
    return ''.join(f'{k} is {v}.' for k, v in x.items()).replace(',.', '. ')

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

class EntityLinkDatasetCollateFnV2:
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

        batch_positive_inputs = self.multi_processor(text=batch_positive,
                                                    images=batch_img,
                                                    return_tensors="pt",
                                                    padding="max_length",
                                                    max_length=self.config.abs_max_length,
                                                    truncation=True)

        return multi_input, batch_mention_token_pos, batch_positive_inputs, batch_negative_inputs

class EntityLinkDatasetCollateFnV3ForEval:
    def __init__(self, multi_processor: FlavaProcessor,
                 config: EntityLinkDatasetConfig,
                 dataset:EntityLinkDataset,
                 callback_num:int=32) -> None:
        self.multi_processor = multi_processor
        self.config = config
        self.dataset=dataset.data
        self.callback_num=callback_num
    def __call__(self, sample):     
        sentence=sample[0]['sentence']
        positive=sample[0]['positive_abstract']
        img=Image.open(sample[0]['image']).convert('RGB')
        
        candidate_data=random.sample(self.dataset,self.callback_num)
        candidates=[]
        for d in candidate_data:
            candidates.append(d['negative_abstract'][0])
        if positive in candidates:
            candidates.remove(positive)
        else:
            candidates.pop()
        candidates.insert(0,positive)
        question_pair=[[sentence,candidates[i]] for i in range(len(candidates))]
        multi_input = self.multi_processor(text=question_pair,
                                           images=[img for _ in range(len(question_pair))],
                                           return_tensors="pt",
                                           padding="max_length",
                                           max_length=160,
                                           truncation=True)
        return multi_input
        
class EntityLinkDatasetCollateFnV3:
    def __init__(self, multi_processor: FlavaProcessor,
                 config: EntityLinkDatasetConfig) -> None:
        self.multi_processor = multi_processor
        self.config = config

    def __call__(self, batch_samples):
        batch_sentence, batch_mention, batch_img, batch_positive, batch_negative = [],[],[],[],[]

        for sample in batch_samples:
            batch_sentence.append(sample['sentence'])
            batch_mention.append(sample['mention'])
            batch_img.append(Image.open(sample['image']).convert('RGB'))
            batch_positive.append(sample['positive_abstract'])
            batch_negative.append(sample['negative_abstract'][0])
        random.shuffle(batch_negative)
        positive_pair=[[batch_sentence[i],batch_positive[i]] for i in range(len(batch_sentence))]
        negative_pair=[[batch_sentence[i],batch_negative[i]] for i in range(len(batch_sentence))]
        multi_input = self.multi_processor(text=positive_pair+negative_pair,
                                           images=batch_img*2,
                                           return_tensors="pt",
                                           padding="max_length",
                                           max_length=160,
                                           truncation=True)
        labels=torch.cat([torch.zeros(len(positive_pair),dtype=int),torch.ones(len(negative_pair),dtype=int)])
        return multi_input,labels

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

def getEntityLinkDataLoaderV2(config: EntityLinkDatasetConfig = EntityLinkDatasetConfig()) -> Tuple[
        DataLoaderX, DataLoaderX]:
    train_dataset = EntityLinkDataset(config.train_MEL_path, config.KG_path, config.img_path)
    val_dataset = EntityLinkDataset(config.val_MEL_path, config.KG_path, config.img_path)
    multi_processor = FlavaProcessor.from_pretrained('facebook/flava-full')
    bert_processor = BertTokenizerFast.from_pretrained('bert-base-cased')
    collateFn = EntityLinkDatasetCollateFnV2(multi_processor, bert_processor, config)
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

def getEntityLinkDataLoaderV3(config: EntityLinkDatasetConfig = EntityLinkDatasetConfig()) -> Tuple[
        DataLoaderX, DataLoaderX,DataLoaderX]:
    train_dataset = EntityLinkDataset(config.train_MEL_path, config.KG_path, config.img_path)
    val_dataset = EntityLinkDataset(config.val_MEL_path, config.KG_path, config.img_path)
    test_dataset = EntityLinkDataset(config.test_MEL_path, config.KG_path, config.img_path)
    multi_processor = FlavaProcessor.from_pretrained('facebook/flava-full')
    collateFn = EntityLinkDatasetCollateFnV3(multi_processor,config)
    collateFnForEval = EntityLinkDatasetCollateFnV3ForEval(multi_processor,config,test_dataset,100)
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
    test_dataloader = DataLoaderX(test_dataset,
                                  batch_size=1,
                                  collate_fn=collateFnForEval,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True)

    torch.manual_seed(config.shuffle_seed)
    torch.cuda.manual_seed_all(config.shuffle_seed)
    np.random.seed(config.shuffle_seed)
    random.seed(config.shuffle_seed)
    torch.backends.cudnn.deterministic = True
    return train_dataloader, val_dataloader,test_dataloader
