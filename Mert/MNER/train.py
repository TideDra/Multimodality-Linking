from accelerate import Accelerator

accelerator = Accelerator()
accelerator.free_memory()
from apex.optimizers import FusedAdam
import logging
from utils import getlogger

logger = getlogger('Mert')
if accelerator.is_main_process:
    logger.info('Loading packages...')
from model import FlavaForNERwithESD_bert_only, FlavaForNERwithESD_bert_blstm
from config import config
from utils import train, evaluate, save_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from transformers import get_scheduler
import warnings
from time import time
from dataset import TwitterDatasetV2, TwitterColloteFnV2, DataLoaderX
import os
#silence logs
warnings.filterwarnings("ignore")
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    if log_name != "Mert" and log_name != 'accelerate':
        log_obj.disabled = True

device = config.device

if __name__ == '__main__':
    writer = None
    if accelerator.is_main_process:
        logger.info('Done.')
        logger.info('Loading model...')

    model = FlavaForNERwithESD_bert_blstm(is_encoder_frozen=True,
                                          is_ESD_encoder_frozen=True)
    model_name = model.__class__.__name__
    #model = model.to(device)
    if accelerator.is_main_process:
        writer = SummaryWriter(os.path.join(config.tb_dir, model_name))
    if accelerator.is_main_process:
        logger.info(
            f'Model:{model_name}|Trained_epoch:{int(model.trained_epoch.item())}'
        )
        logger.info('Done.')
        logger.info('Constructing datasets...')
    train_dataset = TwitterDatasetV2(config.train_text_path,
                                     config.train_img_path, config.batch_size)
    train_dataloader = DataLoaderX(train_dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   collate_fn=TwitterColloteFnV2,
                                   num_workers=config.num_workers,
                                   pin_memory=True)

    dev_dataset = TwitterDatasetV2(config.dev_text_path, config.dev_img_path,
                                   config.batch_size)
    dev_dataloader = DataLoaderX(dev_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 collate_fn=TwitterColloteFnV2,
                                 num_workers=config.num_workers,
                                 pin_memory=True)
    if accelerator.is_main_process:
        logger.info('Done.')
    W_e2n = train_dataset.W_e2n.to(accelerator.device)

    optimizer = FusedAdam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=config.epochs * len(train_dataloader),
    )

    best_micro_f1 = -1.
    if accelerator.is_main_process:
        logger.info('Launching training.')

    model, lr_scheduler, train_dataloader, optimizer = accelerator.prepare(
        model, lr_scheduler, train_dataloader, optimizer)
    dev_dataloader = accelerator.prepare_data_loader(dev_dataloader)
    for epoch in range(config.epochs):
        loss = train(model, train_dataloader, optimizer, lr_scheduler,
                     epoch + 1, W_e2n, writer, accelerator)
        model.trained_epoch += torch.tensor([1],
                                            dtype=torch.float32,
                                            requires_grad=False)
        if accelerator.is_main_process:
            writer.add_scalar('train/epoch_loss', loss, epoch + 1)
        accelerator.print('\n')
        metrics = evaluate(model, dev_dataloader, W_e2n, accelerator)
        valid_macro_f1, valid_micro_f1 = metrics['macro avg'][
            'f1-score'], metrics['micro avg']['f1-score']
        if valid_micro_f1 > best_micro_f1:
            best_micro_f1 = valid_micro_f1
            name = f'{model_name}_epoch_{epoch+1+int(model.trained_epoch.item())}_macrof1_{(100*valid_macro_f1):0.3f}_microf1_{(100*valid_micro_f1):0.3f}_{round(time())}.bin'
            if accelerator.is_main_process:
                save_model(model, name, accelerator)
        accelerator.print(
            '-----------------------------------------------------------')
    if accelerator.is_main_process:
        writer.close()
    if accelerator.is_main_process:
        logger.info('Training over.')
