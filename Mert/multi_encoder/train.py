import os
import sys
from pathlib import Path

from accelerate import Accelerator

accelerator = Accelerator()
accelerator.free_memory()

import logging

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(os.getcwd())

from MNER.utils import getlogger

logger = getlogger('Mert')
if accelerator.is_main_process:
    logger.info('Loading packages...')

import warnings

import torch
import transformers
from torch.utils.tensorboard import SummaryWriter
from transformers.trainer_utils import SchedulerType

from datasets.flickr_dataset import getFlickrDataLoader
from multi_encoder.config import MultiEncoderConfig
from multi_encoder.model import MultiEncoder, MultiEncoderV2_2, MultiEncoderOutput
from multi_encoder.train_config import MultiEncoderTrainConfig
from multi_encoder.train_utils import evaluate, load_model_best, save_model, train

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
    encoder = MultiEncoderV2_2(config=multi_config)
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

    train_config = MultiEncoderTrainConfig()
    start_epoch = 0
    if train_config.load_ckpt:
        ckpt = load_model_best(train_config, model)
        if ckpt is not None:
            start_epoch = ckpt["epoch"]
        ckpt = None
    logger.info(f"start-epoch: {start_epoch}")
    logger.info(f"device: {accelerator.device}")

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
    for epoch in range(start_epoch, train_config.epochs):
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
