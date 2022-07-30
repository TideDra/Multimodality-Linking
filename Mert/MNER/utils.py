from torch.utils.data import Dataset
from config import config
import numpy as np
from torch import tensor
from transformers import AutoTokenizer
from PIL import Image

processor=config.processor

class TwitterDataset(Dataset):
    def __init__(self, file_path:str, img_path:str) -> None:
        self.data = self.load_data(file_path,img_path)
        

    def load_data(self, file_path:str,img_path:str):
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
                    'imgid': img_path+img_id+'.jpg',
                    'sentence': sentence,
                    'tags': tags,
                    'ESD_tags': ESD_tags
                })
            else:
                ind += 1
        # normalize W
        self.W_e2n = self.W_e2n / (self.W_e2n.sum(1).reshape(-1, 1))
        self.W_e2n = tensor(self.W_e2n)
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def TwitterColloteFn(batch_samples):
    batch_sentences, batch_img_inputs = [], []
    for sample in batch_samples:
        batch_sentences.append(sample['sentence'])
        batch_img_inputs.append(Image.open(sample['imgid']))
 
    batch_inputs=processor(text=batch_sentences,images=batch_img_inputs,return_tensors="pt", padding="max_length", max_length=config.max_length,truncation=True)
    batch_tags = np.zeros(shape=batch_inputs['input_ids'].shape,
                          dtype=int)
    batch_ESD_tags = np.zeros(shape=batch_inputs['input_ids'].shape,
                              dtype=int)
    for idx, sentence in enumerate(batch_sentences):
        encoding = processor(text=sentence, truncation=True,return_tensors="pt", padding="max_length", max_length=config.max_length)
        batch_tags[idx][0] = config.special_token_tagid
        batch_tags[idx][len(encoding.tokens()) -
                        1:] = config.special_token_tagid
        batch_ESD_tags[idx][0] = config.special_token_tagid
        batch_ESD_tags[idx][len(encoding.tokens()) -
                            1:] = config.special_token_tagid
        for i in range(1, len(encoding.tokens()) - 1):
            char_start, char_end = encoding.token_to_chars(i)
            tag = batch_samples[idx]['tags'][char_start]
            batch_tags[idx][i] = tag
            ESD_tag = batch_samples[idx]['ESD_tags'][char_start]
            batch_ESD_tags[idx][i] = ESD_tag
    return batch_inputs, tensor(batch_tags), tensor(
        batch_ESD_tags)
