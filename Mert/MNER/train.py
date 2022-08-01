from accelerate import Accelerator
accelerator=Accelerator()
from apex.optimizers import FusedAdam
import logging
from utils import getlogger
logger=getlogger('Mert')
logger.info('Loading packages...')
from model import FlavaForNERwithESD_bert_only,FlavaForNERwithESD_bert_blstm
from config import config
from utils import TwitterDataset, TwitterColloteFn, train, evaluate,save_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from transformers import get_scheduler
import warnings
from time import time
#silence logs
warnings.filterwarnings("ignore")
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    if log_name!="Mert":
          log_obj.disabled = True
    
device = config.device

if __name__ == '__main__':

    logger.info('Done.')
    writer=SummaryWriter(config.tb_dir)
    logger.info('Loading model...')
    model = FlavaForNERwithESD_bert_blstm()
    model_name=model.__class__.__name__
    #model = model.to(device)
    logger.info(f'Model:{model_name}|Trained_epoch:{int(model.trained_epoch.item())}')
    logger.info('Done.')
    logger.info('Constructing datasets.')
    train_dataset = TwitterDataset(config.train_text_path,
                                   config.train_img_path)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  collate_fn=TwitterColloteFn,
                                  num_workers=config.num_workers)

    dev_dataset = TwitterDataset(config.dev_text_path, config.dev_img_path)
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                collate_fn=TwitterColloteFn,
                                num_workers=config.num_workers)
    logger.info('Done.')
    W_e2n = train_dataset.W_e2n.to(accelerator.device)
    
    optimizer = FusedAdam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config.epochs * len(train_dataloader),
    )

    best_macro_f1 = -1.
    best_micro_f1 = -1.
    
    logger.info('Launching training.')

    model,lr_scheduler,train_dataloader,optimizer=accelerator.prepare(model,lr_scheduler,train_dataloader,optimizer)
    dev_dataloader = accelerator.prepare_data_loader(dev_dataloader)
    for epoch in range(config.epochs):
        loss=train(model, train_dataloader, optimizer, lr_scheduler, epoch + 1,
              W_e2n,writer,accelerator)
        model.trained_epoch+=torch.tensor([1],dtype=torch.float32,requires_grad=False)
        writer.add_scalar('train/epoch_loss',loss,epoch+1)
        accelerator.print('\n')
        metrics=evaluate(model,dev_dataloader,W_e2n,accelerator)
        valid_macro_f1, valid_micro_f1 = metrics['macro avg']['f1-score'], metrics['micro avg']['f1-score']
        save_weights = False
        if valid_macro_f1 > best_macro_f1:
            best_macro_f1 = valid_macro_f1
            name=f'{model_name}_epoch_{epoch+1}_macrof1_{(100*valid_macro_f1):0.3f}_microf1_{(100*valid_micro_f1):0.3f}_{round(time())}.bin'
            save_model(model,name,accelerator)
            save_weights = True
        if valid_micro_f1 > best_micro_f1:
            best_micro_f1 = valid_micro_f1
            if not save_weights: 
                name=f'{model_name}_epoch_{epoch+1+int(model.trained_epoch.item())}_macrof1_{(100*valid_macro_f1):0.3f}_microf1_{(100*valid_micro_f1):0.3f}_{round(time())}.bin'
                save_model(model,name,accelerator)
        accelerator.print('-----------------------------------------------------------')
    writer.close()
    logger.info('Training over.')
