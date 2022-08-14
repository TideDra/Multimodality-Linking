import os
import sys

from accelerate import Accelerator

from .train_config import MultiEncoderTrainConfig

accelerator = Accelerator()
accelerator.free_memory()

import logging

sys.path.append(os.getcwd())

from Mert.MNER.utils import getlogger

logger = getlogger('Mert')
if accelerator.is_main_process:
    logger.info('Loading packages...')

import warnings

import torch
import transformers
from Mert.datasets.flickr_dataset import getFlickrDataLoader
from Mert.multi_encoder.config import MultiEncoderConfig
from Mert.multi_encoder.model import MultiEncoder, MultiEncoderOutput
from Mert.multi_encoder.train_utils import evaluate, save_model, train
from torch.utils.tensorboard import SummaryWriter
from transformers.trainer_utils import SchedulerType

warnings.filterwarnings("ignore")
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    if log_name != "Mert" and log_name != 'accelerate':
        log_obj.disabled = True

if __name__ == '__main__':
    writer = None
    if accelerator.is_main_process:
        logger.info('Done.')
        logger.info('Loading model...')

    multi_config = MultiEncoderConfig(d_text=64, d_vision=197)
    encoder = MultiEncoder(config=multi_config)
    model = MultiEncoderOutput(encoder)
    model_name = model.__class__.__name__

    if accelerator.is_main_process:
        writer = SummaryWriter(os.path.join("Mert/multi_encoder/tb_log", model_name))
    if accelerator.is_main_process:
        #logger.info(f'Model:{model_name}|Trained_epoch:{int(model.trained_epoch.item())}')
        logger.info('Done.')
        logger.info('Constructing datasets...')

    train_dl, val_dl = getFlickrDataLoader()

    if accelerator.is_main_process:
        logger.info('Done.')

    train_config=MultiEncoderTrainConfig()

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    lr_scheduler = transformers.get_scheduler(
        SchedulerType.LINEAR,
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=train_config.epochs * len(train_dl),
    )

    best_micro_f1 = -1.
    if accelerator.is_main_process:
        logger.info('Launching training.')

    model, lr_scheduler, train_dl, val_dl, optimizer = accelerator.prepare(
        model, lr_scheduler, train_dl, val_dl, optimizer
    )

    best_loss = 1e6
    for epoch in range(train_config.epochs):
        loss = train(model, train_dl, optimizer, lr_scheduler, epoch + 1, train_config, writer, accelerator)
        if accelerator.is_main_process:
            writer.add_scalar('train/epoch_loss', loss, epoch + 1)
        accelerator.print('\n')
        eval_loss = evaluate(model, val_dl, train_config, accelerator)
        print(f"Eval loss {eval_loss:.8f}")
        if eval_loss < best_loss:
            best_loss = eval_loss
            if accelerator.is_main_process:
                save_model(model, train_config.ckpt_name, epoch + 1, train_config, accelerator)
        accelerator.print('-----------------------------------------------------------')

    if accelerator.is_main_process:
        writer.close()
    if accelerator.is_main_process:
        logger.info('Training over.')
