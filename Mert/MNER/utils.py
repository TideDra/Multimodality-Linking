from torch.utils.data import Dataset
from config import config
import numpy as np
from torch import float32, tensor
from transformers import AutoTokenizer
from PIL import Image
import torch
from tqdm import tqdm
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import os
import sys
import logging

processor = config.processor


class TwitterDataset(Dataset):
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
                    'imgid': img_path + img_id + '.jpg',
                    'sentence': sentence,
                    'tags': tags,
                    'ESD_tags': ESD_tags
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
    batch_sentences, batch_img_inputs = [], []
    for sample in batch_samples:
        batch_sentences.append(sample['sentence'])
        img = Image.open(sample['imgid'])
        batch_img_inputs.append(img.convert('RGB'))

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
    return batch_inputs, tensor(batch_tags), tensor(batch_ESD_tags)


def train(model, dataloader, optimizer, lr_scheduler, epoch, W_e2n, writer,
          accelerator):
    model.train()
    total_loss = 0.0
    with tqdm(enumerate(dataloader, start=1),
              unit='batch',
              total=len(dataloader),
              desc='epoch:{}/{}'.format(epoch, config.epochs),
              disable=not accelerator.is_local_main_process) as tbar:
        for idx, (inputs, labels, ESD_labels) in tbar:
            #inputs = inputs.to(config.device)
            #labels = labels.to(config.device)
            #ESD_labels = ESD_labels.to(config.device)
            _, loss = model(inputs, W_e2n, labels, ESD_labels)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            loss=accelerator.gather(loss)
            total_loss += loss.item()
            if accelerator.is_main_process:
                writer.add_scalar('train/batch_loss', loss.item(),
                              len(dataloader) * (epoch - 1) + idx)
            tbar.set_postfix(loss="%.2f" %
                             ((total_loss / idx) / config.batch_size))
            tbar.update()
    return total_loss


def evaluate(model, dataloader, W_e2n, accelerator):
    model.eval()
    pred_tags = []
    true_tags = []
    with torch.no_grad():
        for inputs, labels, ESD_labels in tqdm(dataloader,
                                               unit='batch',
                                               total=len(dataloader) + 1,
                                               desc='Evaluating...'):
            #inputs = inputs.to(config.device)
            logits, _ = model(inputs, W_e2n)
            logits = accelerator.gather(logits)
            inputs = accelerator.gather(inputs)
            labels = accelerator.gather(labels)
            mask = inputs['attention_mask'].bool()
            pred_labels = model.crf.decode(logits, mask)
            true_labels = labels.tolist()

            pred_tags += [[config.id2tag[id] for id in seq]
                          for seq in pred_labels]
            true_tags += [[config.id2tag[id] for id, m in zip(seq, mask) if m]
                          for seq, mask in zip(true_labels, mask.tolist())]
    print(
        classification_report(true_tags, pred_tags, mode='strict',
                              scheme=IOB2))
    return classification_report(true_tags,
                                 pred_tags,
                                 mode='strict',
                                 scheme=IOB2,
                                 output_dict=True)


def save_model(model, name,accelerator):
    accelerator.print('Saving checkpoint...\n')
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), os.path.join(config.checkpoint_path, name))
    checkpoint_list = os.listdir(config.checkpoint_path)
    if (len(checkpoint_list) > config.max_checkpoint_num):
        file_map = {}
        times = []
        del_num = len(checkpoint_list) - config.max_checkpoint_num
        for f in checkpoint_list:
            t = f.split('.')[0].split('_')[-1]
            file_map[int(t)] = os.path.join(config.checkpoint_path, f)
            times.append(int(t))
        times.sort()
        for i in range(del_num):
            del_f = file_map[times[i]]
            os.remove(del_f)
    accelerator.print('Checkpoint has been updated successfully.\n')


def getlogger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger