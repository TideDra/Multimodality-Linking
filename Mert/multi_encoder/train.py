import logging
import os

import torch
import transformers
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers.trainer_utils import SchedulerType

from .model import MultiEncoderOutput
from .train_config import MultiEncoderTrainConfig
from .train_utils import evaluate, load_model_best, save_model, train


def main_train(accelerator: Accelerator, logger: logging.Logger):
    writer = None

    train_config = MultiEncoderTrainConfig()

    if accelerator.is_main_process:
        logger.info('Done.')
        logger.info('Loading model...')

    model = MultiEncoderOutput(train_config.encoder)
    model_name = model.__class__.__name__

    if accelerator.is_main_process:
        writer = SummaryWriter(os.path.join("Mert/multi_encoder/tb_log", model_name))
    if accelerator.is_main_process:
        logger.info('Done.')
        logger.info('Constructing datasets...')

    train_dl, val_dl = train_config.dataloader

    if accelerator.is_main_process:
        logger.info('Done.')

    start_epoch = 0
    if train_config.load_ckpt:
        try: # 这个不能起作用
            ckpt = load_model_best(train_config, model)
            if ckpt is not None:
                start_epoch = ckpt["epoch"]
            ckpt = None
        except Exception as e:
            logger.error("Fail to load checkpoint.\n" + e)

    logger.info(f"start-epoch: {start_epoch}")
    logger.info(f"device: {accelerator.device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    lr_scheduler = transformers.get_scheduler(
        SchedulerType.LINEAR,
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=train_config.epochs * len(train_dl),
    )

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
                save_model(model, epoch + 1, train_config, accelerator)
        accelerator.print('-----------------------------------------------------------')

    if accelerator.is_main_process:
        writer.close()
    if accelerator.is_main_process:
        logger.info('Training over.')
