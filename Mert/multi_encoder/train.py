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
from Mert.multi_encoder.config import MultiEncoderConfig
from Mert.multi_encoder.model import MultiEncoder, MultiEncoderOutput
from Mert.multi_encoder.train_utils import train
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
    encoder = MultiEncoder(fusion_config=multi_config)
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
        file_path="Mert/MNER/data/Twitter2015/train.txt",
        img_path="Mert/MNER/data/Twitter2015_images",
        batch_size=config.batch_size,
    )
    train_dl = DataLoaderX(
        train_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=TwitterColloteFnV2,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # val_ds = TwitterDatasetV2(
    #     file_path="Mert/MNER/data/Twitter2015/valid.txt",
    #     img_path="Mert/MNER/data/Twitter2015_images",
    #     batch_size=config.batch_size
    # )
    # val_ds = DataLoaderX(
    #     val_ds,
    #     batch_size=1,
    #     shuffle=False,
    #     collate_fn=TwitterColloteFnV2,
    #     num_workers=config.num_workers,
    #     pin_memory=True
    # )
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

    model, lr_scheduler, train_dl, optimizer = accelerator.prepare(model, lr_scheduler, train_dl, optimizer)
    #val_ds = accelerator.prepare_data_loader(val_ds)
    for epoch in range(config.epochs):
        loss = train(model, train_dl, optimizer, lr_scheduler, epoch + 1, multi_config, writer, accelerator)
        #model.trained_epoch += torch.tensor([1], dtype=torch.float32, requires_grad=False)
        if accelerator.is_main_process:
            writer.add_scalar('train/epoch_loss', loss, epoch + 1)
        accelerator.print('\n')
        # metrics = evaluate(model, val_ds, W_e2n, accelerator)
        # valid_macro_f1, valid_micro_f1 = metrics['macro avg']['f1-score'], metrics['micro avg']['f1-score']
        # if valid_micro_f1 > best_micro_f1:
        #     best_micro_f1 = valid_micro_f1
        #     name = f'{model_name}_epoch_{epoch+1+int(model.trained_epoch.item())}_macrof1_{(100*valid_macro_f1):0.3f}_microf1_{(100*valid_micro_f1):0.3f}_{round(time())}.bin'
        #     #if accelerator.is_main_process:
        #     #    save_model(model, name, accelerator)
        accelerator.print('-----------------------------------------------------------')
    if accelerator.is_main_process:
        writer.close()
    if accelerator.is_main_process:
        logger.info('Training over.')
