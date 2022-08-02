from torch.utils.data import Dataset, DataLoader
from config import config
import numpy as np
from PIL import Image
import os
from torch import tensor, float32
from prefetch_generator import BackgroundGenerator

processor = config.processor


class DataLoaderX(DataLoader):
    '''
    A replacement to DataLoader which improves the pipeline performance.
    '''
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class TwitterDataset(Dataset):
    '''
    Dataset for Twitter2015 and Twitter2017
    '''
    def __init__(self, file_path: str, img_path: str) -> None:
        self.data = self.load_data(file_path, img_path)

    def load_data(self, file_path: str, img_path: str):
        dataset = open(file_path).readlines()
        Data = []
        ind = 0
        end = len(dataset)
        self.W_e2n = np.zeros((len(config.ESD_tag2id), len(config.tag2id)))
        while ind < end:
            text = dataset[ind]
            if text[:5] == 'IMGID':
                img_id = text[6:-1]
                ind += 1
                if dataset[ind][:2] == 'RT':  # skip name
                    ind += 3
                # read sentence
                sentence = ''
                tags = []
                ESD_tags = []
                while ind < end:
                    text = dataset[ind]
                    if text == '\n':  # reach the end of a sample
                        ind += 1
                        break
                    if text[:4] == 'http':  # skip url
                        ind += 1
                        continue
                    word, tag = text.replace('\n', '').split('\t')
                    if sentence != '':  # use space to split words
                        sentence += ' '
                        tags.append(0)
                        ESD_tags.append(0)
                    sentence += word
                    # tag is mapped to char
                    tags += [config.tag2id[tag]] * len(word)
                    ESD_tags += [config.ESD_tag2id[tag[0]]] * len(word)
                    self.W_e2n[config.ESD_tag2id[tag[0]]][
                        config.tag2id[tag]] += 1
                    ind += 1
                Data.append({
                    'img':
                    Image.open(os.path.join(img_path, img_id) +
                               '.jpg').convert('RGB'),
                    'sentence':
                    sentence,
                    'tags':
                    tags,
                    'ESD_tags':
                    ESD_tags
                })
            else:
                ind += 1
        # normalize W
        self.W_e2n = self.W_e2n / (self.W_e2n.sum(1).reshape(-1, 1))
        self.W_e2n = tensor(self.W_e2n, requires_grad=False, dtype=float32)
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def TwitterColloteFn(batch_samples):
    '''
    Collate function for TwitterDataset. This Collate function contains too many CPU and IO operation, so it shouldn't be used 
    as train dataloader(dataset), especially in multi-GPU training.
    '''
    batch_sentences, batch_img_inputs = [], []
    for sample in batch_samples:
        batch_sentences.append(sample['sentence'])
        batch_img_inputs.append(sample['img'])

    batch_inputs = processor(text=batch_sentences,
                             images=batch_img_inputs,
                             return_tensors="pt",
                             padding="max_length",
                             max_length=config.max_length,
                             truncation=True)
    batch_tags = np.zeros(shape=batch_inputs['input_ids'].shape, dtype=int)
    batch_ESD_tags = np.zeros(shape=batch_inputs['input_ids'].shape, dtype=int)
    for idx, sentence in enumerate(batch_sentences):
        encoding = processor(text=sentence,
                             truncation=True,
                             return_tensors="pt",
                             padding="max_length",
                             max_length=config.max_length)
        SEP_pos = encoding.tokens().index('[SEP]')
        batch_tags[idx][0] = config.special_token_tagid
        batch_tags[idx][SEP_pos:] = config.special_token_tagid
        batch_ESD_tags[idx][0] = config.special_token_tagid
        batch_ESD_tags[idx][SEP_pos:] = config.special_token_tagid
        for i in range(1, SEP_pos):
            char_start, char_end = encoding.token_to_chars(i)
            tag = batch_samples[idx]['tags'][char_start]
            batch_tags[idx][i] = tag
            ESD_tag = batch_samples[idx]['ESD_tags'][char_start]
            batch_ESD_tags[idx][i] = ESD_tag

    return batch_inputs, tensor(batch_tags), tensor(
        batch_ESD_tags), batch_sentences


class TwitterDatasetV2(Dataset):
    '''
    A pre-process version of TwitterDataset. this implementation finish all the pre-process operations during constructing the dataset, e.g. tokenize, Image IO.
    We strongly recommend to use this version, especially in multi-GPU training, which efficiently decrease the waiting time of GPUs.
    '''
    def __init__(self, file_path: str, img_path: str, batch_size: int) -> None:
        '''
        Args:
            file_path(str): path of text data file.
            img_path(str): path of image data folder.
            batch_size(int): Different from common dataset, you need to define the batch_size when constructing the dataset.
        Because it consumes too much time to construct a batch(see the TwitterColloteFn), we don't let dataloader to do this, which will bolck GPUs.
        Consequently, you must set the batch_size of dataloader to 1. A sample from the dataset is already a batch. To avoid dataloader batch the sample
        again, you also need to use TwitterCollateFnV2 as the ''collate_fn'' of dataloader.
        '''
        self.data = self.load_data(file_path, img_path)
        self.processor = config.processor
        self.batch_data = self.get_batch_data(batch_size)

    def load_data(self, file_path: str, img_path: str):
        dataset = open(file_path).readlines()
        Data = []
        ind = 0
        end = len(dataset)
        self.W_e2n = np.zeros((len(config.ESD_tag2id), len(config.tag2id)))
        while ind < end:
            text = dataset[ind]
            if text[:5] == 'IMGID':
                img_id = text[6:-1]
                ind += 1
                if dataset[ind][:2] == 'RT':  # skip name
                    ind += 3
                # read sentence
                sentence = ''
                tags = []
                ESD_tags = []
                while ind < end:
                    text = dataset[ind]
                    if text == '\n':  # reach the end of a sample
                        ind += 1
                        break
                    if text[:4] == 'http':  # skip url
                        ind += 1
                        continue
                    word, tag = text.replace('\n', '').split('\t')
                    if sentence != '':  # use space to split words
                        sentence += ' '
                        tags.append(0)
                        ESD_tags.append(0)
                    sentence += word
                    # tag is mapped to char
                    tags += [config.tag2id[tag]] * len(word)
                    ESD_tags += [config.ESD_tag2id[tag[0]]] * len(word)
                    self.W_e2n[config.ESD_tag2id[tag[0]]][
                        config.tag2id[tag]] += 1
                    ind += 1
                Data.append({
                    'img':
                    Image.open(os.path.join(img_path, img_id) +
                               '.jpg').convert('RGB'),
                    'sentence':
                    sentence,
                    'tags':
                    tags,
                    'ESD_tags':
                    ESD_tags
                })
            else:
                ind += 1
        # normalize W
        self.W_e2n = self.W_e2n / (self.W_e2n.sum(1).reshape(-1, 1))
        self.W_e2n = tensor(self.W_e2n, requires_grad=False, dtype=float32)
        return Data

    def get_batch_data(self, batch_size):
        batch_data = []
        for i in range(0, len(self.data), batch_size):
            batch_samples = []
            for k in range(i, min(i + batch_size, len(self.data))):
                batch_samples.append(self.data[k])
            batch_sentences, batch_img_inputs = [], []
            for sample in batch_samples:
                batch_sentences.append(sample['sentence'])
                batch_img_inputs.append(sample['img'])

            batch_inputs = self.processor(text=batch_sentences,
                                          images=batch_img_inputs,
                                          return_tensors="pt",
                                          padding="max_length",
                                          max_length=config.max_length,
                                          truncation=True)
            batch_tags = np.zeros(shape=batch_inputs['input_ids'].shape,
                                  dtype=int)
            batch_ESD_tags = np.zeros(shape=batch_inputs['input_ids'].shape,
                                      dtype=int)
            for idx, sentence in enumerate(batch_sentences):
                encoding = self.processor(text=sentence,
                                          truncation=True,
                                          return_tensors="pt",
                                          padding="max_length",
                                          max_length=config.max_length)
                SEP_pos = encoding.tokens().index('[SEP]')
                batch_tags[idx][0] = config.special_token_tagid
                batch_tags[idx][SEP_pos:] = config.special_token_tagid
                batch_ESD_tags[idx][0] = config.special_token_tagid
                batch_ESD_tags[idx][SEP_pos:] = config.special_token_tagid
                for i in range(1, SEP_pos):
                    char_start, char_end = encoding.token_to_chars(i)
                    tag = batch_samples[idx]['tags'][char_start]
                    batch_tags[idx][i] = tag
                    ESD_tag = batch_samples[idx]['ESD_tags'][char_start]
                    batch_ESD_tags[idx][i] = ESD_tag
            batch_data.append({
                'batch_inputs': batch_inputs,
                'batch_tags': tensor(batch_tags),
                'batch_ESD_tags': tensor(batch_ESD_tags)
            })
        return batch_data

    def __len__(self):
        return len(self.batch_data)

    def __getitem__(self, idx):
        return self.batch_data[idx]


def TwitterColloteFnV2(batch_sample):
    '''
    The default ''collate_fn'' of dataloader will batch the sample again, which is already a batch. Use this collate_fn to avoid it.
    '''
    sample = batch_sample[0]
    return sample['batch_inputs'], sample['batch_tags'], sample[
        'batch_ESD_tags']
