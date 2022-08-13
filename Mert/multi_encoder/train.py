import sys
import os

sys.path.append(os.getcwd())

from accelerate import Accelerator

accelerator = Accelerator()
accelerator.free_memory()

import logging

from Mert.MNER.utils import getlogger

logger = getlogger('Mert')
if accelerator.is_main_process:
    logger.info('Loading packages...')

import warnings

import torch
import transformers
from Mert.MNER.config import config
from Mert.MNER.dataset import DataLoaderX, TwitterColloteFnV2, TwitterDatasetV2
from Mert.multi_encoder.config import MultiEncoderConfig, TwitterDatasetTrainConfig
from Mert.multi_encoder.model import MultiEncoder, MultiEncoderOutput
from Mert.multi_encoder.train_utils import evaluate, save_model, train
from torch.utils.tensorboard import SummaryWriter
from transformers.trainer_utils import SchedulerType

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

    multi_config = MultiEncoderConfig(d_text=config.max_length, d_vision=197)
    encoder = MultiEncoder(config=multi_config)
    model = MultiEncoderOutput(encoder)
    model_name = model.__class__.__name__
    #model = model.to(device)
    if accelerator.is_main_process:
        writer = SummaryWriter(os.path.join("Mert/multi_encoder/tb_log", model_name))
    if accelerator.is_main_process:
        #logger.info(f'Model:{model_name}|Trained_epoch:{int(model.trained_epoch.item())}')
        logger.info('Done.')
        logger.info('Constructing datasets...')

    train_ds = TwitterDatasetV2(
        batch_size=config.batch_size,
        file_path=TwitterDatasetTrainConfig.train_file_path,
        img_path=TwitterDatasetTrainConfig.train_img_path,
        cache_path=TwitterDatasetTrainConfig.train_cache_path
    )
    train_dl = DataLoaderX(
        train_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=TwitterColloteFnV2,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_ds = TwitterDatasetV2(
        batch_size=config.batch_size,
        file_path=TwitterDatasetTrainConfig.val_file_path,
        img_path=TwitterDatasetTrainConfig.val_img_path,
        cache_path=TwitterDatasetTrainConfig.val_cache_path
    )
    val_dl = DataLoaderX(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=TwitterColloteFnV2,
        num_workers=config.num_workers,
        pin_memory=True
    )

    if accelerator.is_main_process:
        logger.info('Done.')

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = transformers.get_scheduler(
        SchedulerType.LINEAR,
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=config.epochs * len(train_dl),
    )

    best_micro_f1 = -1.
    if accelerator.is_main_process:
        logger.info('Launching training.')

    model, lr_scheduler, train_dl, val_dl, optimizer = accelerator.prepare(
        model, lr_scheduler, train_dl, val_dl, optimizer
    )

    best_loss = 1e6
    for epoch in range(config.epochs):
        loss = train(model, train_dl, optimizer, lr_scheduler, epoch + 1, multi_config, writer, accelerator)
        #model.trained_epoch += torch.tensor([1], dtype=torch.float32, requires_grad=False)
        if accelerator.is_main_process:
            writer.add_scalar('train/epoch_loss', loss, epoch + 1)
        accelerator.print('\n')
        eval_loss = evaluate(model, val_dl, multi_config, accelerator)
        print(f"Eval loss {eval_loss:.8f}")
        if eval_loss < best_loss:
            best_loss = eval_loss
            if accelerator.is_main_process:
                save_model(model, "multi-encoder", epoch + 1, multi_config, accelerator)
        accelerator.print('-----------------------------------------------------------')
    if accelerator.is_main_process:
        writer.close()
    if accelerator.is_main_process:
        logger.info('Training over.')
