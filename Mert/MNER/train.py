import logging
from model import FlavaForNERwithESD_bert_only
from config import config
from utils import TwitterDataset, TwitterColloteFn, train, evaluate,save_model,getlogger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from transformers import get_scheduler
import warnings
from time import time
#silence logs
warnings.filterwarnings("ignore")
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
          log_obj.disabled = True
    
device = config.device

if __name__ == '__main__':

    logger=getlogger('Logger')
    writer=SummaryWriter(config.tb_dir)
    logger.info('Loading model...')
    model = FlavaForNERwithESD_bert_only()

    model = model.to(device)
    logger.info('Model loaded successfully.')
    logger.info('Constructing datasets.')
    train_dataset = TwitterDataset(config.train_text_path,
                                   config.train_img_path)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  collate_fn=TwitterColloteFn)

    dev_dataset = TwitterDataset(config.dev_text_path, config.dev_img_path)
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                collate_fn=TwitterColloteFn)
    logger.info('Datasets constructed successfully.')
    W_e2n = train_dataset.W_e2n.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config.epochs * len(train_dataloader),
    )

    best_macro_f1 = -1.
    best_micro_f1 = -1.

    logger.info('Launching training.')
    for epoch in range(config.epochs):
        loss=train(model, train_dataloader, optimizer, lr_scheduler, epoch + 1,
              W_e2n,writer)
        writer.add_scalar('train/epoch_loss',loss,epoch+1)
        print('\n')
        metrics=evaluate(model,dev_dataloader,W_e2n)
        valid_macro_f1, valid_micro_f1 = metrics['macro avg']['f1-score'], metrics['micro avg']['f1-score']
        save_weights = False
        if valid_macro_f1 > best_macro_f1:
            best_macro_f1 = valid_macro_f1
            name=f'epoch_{epoch+1}_macrof1_{(100*valid_macro_f1):0.3f}_microf1_{(100*valid_micro_f1):0.3f}_{round(time())}.bin'
            save_model(model,name)
            save_weights = True
        if valid_micro_f1 > best_micro_f1:
            best_micro_f1 = valid_micro_f1
            if not save_weights: 
                name=f'epoch_{epoch+1}_macrof1_{(100*valid_macro_f1):0.3f}_microf1_{(100*valid_micro_f1):0.3f}_{round(time())}.bin'
                save_model(model,name)
        print('----------------------------------')
    writer.close()
    logger.info('Training over.')
