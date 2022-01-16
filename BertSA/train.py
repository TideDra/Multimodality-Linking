import numpy as np
import torch
import pandas as pd
from torch.nn.utils import clip_grad_norm
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import time


def read_dataset(filename):
    dataset = pd.read_csv(filename, encoding='utf-8')
    _review = dataset['review'].values
    _label = dataset['label'].values
    return _review, _label


def text2token(sentences, limitsize=126):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    final_tokens = []
    for s in sentences:
        s = s[:limitsize]
        s_token = tokenizer.encode(s)
        if len(s_token) < limitsize + 2:
            s_token.extend([0] * (limitsize + 2 - len(s_token)))
        final_tokens.append(s_token)
    return final_tokens


def attention_masks(sent_tokens):
    masks = []
    for i in sent_tokens:
        i_mask = [float(k != 0) for k in i]
        masks.append(i_mask)
    return masks


def accuracy(preds, labels):
    matches = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()
    acc = matches.sum().item() / len(matches)
    return acc


def evaluate(model, test_batches):
    per_acc = []
    model.eval()
    with torch.no_grad():
        for batch in test_batches:
            batch_data, batch_mask, batch_label = batch[0].long().to(device), batch[1].long().to(device), batch[
                2].long().to(device)
            output = model(batch_data, token_type_ids=None, attention_mask=batch_mask, labels=batch_label)
            loss, logits = output[0], output[1]
            per_acc.append(accuracy(logits, batch_label))
        return np.array(per_acc).mean()


testsize = 0.2
random_seed = 777
learning_rate = 2e-5
epoch = 2

review, label = read_dataset('dataset/weibo_senti_100k.csv')
total_labels = torch.tensor(label)
total_reviews = torch.tensor(text2token(review))
atten_masks = torch.tensor(attention_masks(total_reviews))
train_data, test_data, train_label, test_label = train_test_split(total_reviews, total_labels, test_size=testsize,
                                                                  random_state=random_seed)
train_mask, test_mask = train_test_split(atten_masks, test_size=testsize,
                                         random_state=random_seed)
train_inputs = TensorDataset(train_data, train_mask, train_label)
train_sampler = RandomSampler(train_inputs)
train_batches = DataLoader(train_inputs, batch_size=32, shuffle=False, sampler=train_sampler)
test_inputs = TensorDataset(test_data, test_mask, test_label)
test_sampler = RandomSampler(test_inputs)
test_batches = DataLoader(test_inputs, batch_size=32, shuffle=False, sampler=test_sampler)

model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_step = epoch * len(train_batches)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=total_step)

t0 = time.time()
per_acc = []
for e in range(epoch):
    for step, batch in enumerate(train_batches):
        if step % 40 == 0 and step != 0:
            run_time = int(time.time() - t0)
            print("Epoch {}: Batch {} of {}, elapse: {}".format(e, step, len(train_batches), run_time))
        batch_data, batch_mask, batch_label = batch[0].long().to(device), batch[1].long().to(device), batch[
            2].long().to(device)
        torch.cuda.empty_cache()
        output = model(batch_data, token_type_ids=None, attention_mask=batch_mask, labels=batch_label)
        loss, logits = output[0], output[1]
        per_acc.append(accuracy(logits, batch_label))

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    print("{0:-^20}".format("Epoch {} ACC:{}".format(e, np.array(per_acc).mean())))
    torch.save(model, './model_{}.pth'.format(e))
