from model import MertForNERwithESD, MertForNER
from transformers import FlavaModel, BertModel
from config import config
from utils import TwitterDataset, TwitterColloteFn
from torch.utils.data import DataLoader
import torch
from transformers import AdamW, get_scheduler
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = FlavaModel.from_pretrained("facebook/flava-full")
    ESD_encoder = BertModel.from_pretrained('bert-base-cased')
    model = MertForNERwithESD(encoder=encoder,
                              ESD_encoder=ESD_encoder,
                              num_tags=len(config.tag2id),
                              ESD_num_tags=len(config.ESD_tag2id),
                              ratio=1)

    model = model.to(device)
    train_dataset = TwitterDataset(config.train_text_path,
                                   config.train_img_path)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  collate_fn=TwitterColloteFn)

    W_e2n = train_dataset.W_e2n

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config.epochs * len(train_dataloader),
    )

    for epoch in range(config.epochs):
        total_loss = 0
        for idx, inputs, labels, ESD_labels in enumerate(train_dataloader,start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            ESD_labels = ESD_labels.to(device)
            _, loss = model(inputs, labels, ESD_labels, W_e2n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss+=loss.item()
            if idx%100==0:
                print('epoch:{} batch:{} loss;{}'.format(epoch,idx,total_loss/idx))