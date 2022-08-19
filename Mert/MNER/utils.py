from .config import config
import torch
from tqdm import tqdm
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import os
import sys
import logging
import pickle
processor = config.processor


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
            loss = accelerator.gather(loss)
            total_loss += loss.sum().item()
            if accelerator.is_main_process:
                writer.add_scalar('train/batch_loss',
                                  loss.sum().item(),
                                  len(dataloader) * (epoch - 1) + idx)
            tbar.set_postfix(loss="%.2f" %
                             ((total_loss / idx) / config.batch_size))
            tbar.update()
    return total_loss


def evaluate(model, dataloader, W_e2n, accelerator, test_ESD=False):
    '''
    Evaluate the P,R and F1 use the ''seqeval'' package in token level.
    Args:
        model: the model to evaluate.
        dataloader: the dataloader.
        W_e2n: transition matrix used for ESD module.
        acceelerator: accelerator used for distributed training.
        test_ESD: if ''False'', only evaluate ESD. Other wise evaluate NER. 
    '''
    model.eval()
    pred_tags = []
    true_tags = []
    with torch.no_grad():
        for inputs, labels, ESD_labels in tqdm(
                dataloader,
                unit='batch',
                total=len(dataloader) + 1,
                desc='Evaluating...',
                disable=not accelerator.is_local_main_process):
            #inputs = inputs.to(config.device)
            if test_ESD:
                labels = ESD_labels
                id2tag = config.ESD_id2tag
            else:
                id2tag = config.id2tag
            logits, _ = model(inputs, W_e2n)
            logits = accelerator.gather(logits)
            inputs = accelerator.gather(inputs)
            labels = accelerator.gather(labels)
            mask = inputs['attention_mask'].bool()
            pred_labels = model.crf.decode(logits, mask)
            true_labels = labels.tolist()

            pred_tags += [[id2tag[id] for id in seq] for seq in pred_labels]
            true_tags += [[id2tag[id] for id, m in zip(seq, mask) if m]
                          for seq, mask in zip(true_labels, mask.tolist())]
    accelerator.print(
        classification_report(true_tags, pred_tags, mode='strict',
                              scheme=IOB2))
    return classification_report(true_tags,
                                 pred_tags,
                                 mode='strict',
                                 scheme=IOB2,
                                 output_dict=True)


def NERpipeline(model, text=None, img=None, inputs=None, W_e2n=None):
    '''
    Do multi-modal NER for given text and img.
    '''
    if inputs == None:
        if type(text) != type([]):
            text = [text]
        if type(img) != type([]):
            img = [img]
        assert len(text) == len(
            img), "number of texts must equal to number of imgs."
        inputs = config.processor(text=text,
                                  images=img,
                                  return_tensors="pt",
                                  padding="max_length",
                                  max_length=config.max_length,
                                  truncation=True)
    inputs = inputs.to(config.device)
    if W_e2n==None:
        with open('/home/zero_lag/Document/srtp/Multimodality-Link/Mert/MNER/W_e2n','rb') as f:
            W_e2n=pickle.load(f)
    W_e2n = W_e2n.to(config.device)
    model.eval()
    with torch.no_grad():
        logits, _ = model(inputs, W_e2n)
        mask = inputs['attention_mask'].bool()
        pred_labels = model.crf.decode(logits, mask)
    pred_token_tags = [[config.id2tag[id] for id in seq]
                       for seq in pred_labels]

    batch_size = inputs['input_ids'].shape[0]
    result = []
    for b in range(batch_size):
        word_ids = set(inputs.word_ids(b))
        word_ids.remove(None)
        recording = False
        entities = []
        entity_head = None
        entity_type = None
        for word_id in range(len(word_ids)):
            token_start, token_end = inputs.word_to_tokens(b, word_id)
            word_tag = pred_token_tags[b][token_start]
            if word_tag[0] == 'B' and recording == False:
                recording = True
                token_ids = [i for i in range(token_start, token_end)]
                entity_head, _ = inputs.word_to_chars(b, word_id)
                entity_type = word_tag.split('-')[-1]
            elif word_tag[0] == 'B' and recording == True:
                _, last_entity_end = inputs.word_to_chars(b, word_id - 1)
                entities.append({
                    'entity': text[b][entity_head:last_entity_end],
                    'type': entity_type,
                    'token_ids': token_ids
                })
                token_ids = []
                token_ids = [i for i in range(token_start, token_end)]
                entity_head, _ = inputs.word_to_chars(b, word_id)
                entity_type = word_tag.split('-')[-1]
            elif word_tag[0] == 'I' and recording == True:
                token_ids += [i for i in range(token_start, token_end)]
            elif word_tag[0] == 'O' and recording == True:
                _, last_entity_end = inputs.word_to_chars(b, word_id - 1)
                entities.append({
                    'entity': text[b][entity_head:last_entity_end],
                    'type': entity_type,
                    'token_ids': token_ids
                })
                recording = False
                token_ids = []
        result.append({
            'sentence': text[b],
            'entities': entities,
            'image': img[b]
        })
    return result


def evaluate_word_level(model, dataloader, W_e2n, accelerator, test_ESD=False):
    '''
    Evaluate the P,R and F1 use the ''seqeval'' package in word level.
    Args:
        model: the model to evaluate.
        dataloader: the dataloader.
        W_e2n: transition matrix used for ESD module.
        acceelerator: accelerator used for distributed training.
        test_ESD: if ''False'', only evaluate ESD. Other wise evaluate NER. 
    '''
    model.eval()
    pred_tags = []
    true_tags = []
    with torch.no_grad():
        for inputs, labels, ESD_labels, sentences in tqdm(
                dataloader,
                unit='batch',
                total=len(dataloader) + 1,
                desc='Evaluating...',
                disable=not accelerator.is_local_main_process):
            #inputs = inputs.to(config.device)
            if test_ESD:
                labels = ESD_labels
                id2tag = config.ESD_id2tag
            else:
                id2tag = config.id2tag
            logits, _ = model(inputs, W_e2n)
            logits = accelerator.gather(logits)
            inputs = accelerator.gather(inputs)
            labels = accelerator.gather(labels)

            mask = inputs['attention_mask'].bool()
            pred_labels = model.crf.decode(logits, mask)
            true_labels = labels.tolist()
            inputs = config.processor(text=sentences,
                                      truncation=True,
                                      return_tensors="pt",
                                      padding="max_length",
                                      max_length=config.max_length)
            pred_token_tags = [[id2tag[id] for id in seq]
                               for seq in pred_labels]
            true_token_tags = [[id2tag[id] for id, m in zip(seq, mask) if m]
                               for seq, mask in zip(true_labels, mask.tolist())
                               ]

            batch_size = inputs['input_ids'].shape[0]
            for b in range(batch_size):
                word_ids = set(inputs.word_ids(b))
                word_ids.remove(None)
                pred_word_tags = []
                true_word_tags = []
                for word_id in range(len(word_ids)):
                    start, end = inputs.word_to_tokens(b, word_id)
                    pred_word_tags.append(pred_token_tags[b][start])
                    true_word_tags.append(true_token_tags[b][start])
                pred_tags.append(pred_word_tags)
                true_tags.append(true_word_tags)
    print(
        classification_report(true_tags, pred_tags, mode='strict',
                              scheme=IOB2))
    return classification_report(true_tags,
                                 pred_tags,
                                 mode='strict',
                                 scheme=IOB2,
                                 output_dict=True)


def save_model(model, name, accelerator):
    accelerator.print('Saving checkpoint...\n')
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    model_name = name.split('_epoch')[0]
    checkpoint_path = config.checkpoint_path
    if model_name not in os.listdir(checkpoint_path):
        os.mkdir(os.path.join(checkpoint_path, model_name))
    checkpoint_path = os.path.join(checkpoint_path, model_name)
    accelerator.save(unwrapped_model.state_dict(),
                     os.path.join(checkpoint_path, name))
    checkpoint_list = os.listdir(checkpoint_path)
    if (len(checkpoint_list) > config.max_checkpoint_num):
        file_map = {}
        times = []
        del_num = len(checkpoint_list) - config.max_checkpoint_num
        for f in checkpoint_list:
            t = f.split('.')[0].split('_')[-1]
            file_map[int(t)] = os.path.join(checkpoint_path, f)
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
