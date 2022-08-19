import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
from accelerate import Accelerator

accelerator = Accelerator()
accelerator.free_memory()
from apex.optimizers import FusedAdam
import logging

from Datasets.EntityLink_dataset import getEntityLinkDataLoader
logger = getlogger('Mert')
if accelerator.is_main_process:
    logger.info('Loading packages...')
from multi_encoder.model import MultiEncoderV2_2


from torch.utils.tensorboard import SummaryWriter
import torch
from transformers import get_scheduler,AutoModel
import warnings
from time import time
from Recall_entity.config import config
from Recall_entity.utils import train,evaluate,getlogger,save_model
from Recall_entity.loss import TripletLoss
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

    multimodel=MultiEncoderV2_2()
    entity_model=AutoModel.from_pretrained('bert-base-cased')
    model_name = multimodel.__class__.__name__
    #model = model.to(device)
    if accelerator.is_main_process:
        writer = SummaryWriter(os.path.join(config.tb_dir, model_name))
    if accelerator.is_main_process:
        logger.info(
            f'Model:{model_name}'
        )
        logger.info('Done.')
        logger.info('Constructing datasets...')
    
    train_dataloader,val_dataloader=getEntityLinkDataLoader()
    if accelerator.is_main_process:
        logger.info('Done.')

    optimizer = FusedAdam(multimodel.parameters(), lr=config.learning_rate,weight_decay=0.2)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=config.epochs * len(train_dataloader),
    )
    if accelerator.is_main_process:
        logger.info('Launching training.')
    multimodel, lr_scheduler, train_dataloader, optimizer = accelerator.prepare(
        multimodel, lr_scheduler, train_dataloader, optimizer)
    entity_model=accelerator.prepare(entity_model)

    criterion=TripletLoss
    best_acc=0
    for epoch in range(config.epochs):
        loss = train(multimodel, entity_model,criterion,train_dataloader, optimizer, lr_scheduler,
                     epoch + 1, writer, accelerator)
        
        if accelerator.is_main_process:
            writer.add_scalar('train/epoch_loss', loss, epoch + 1)
        accelerator.print('\n')

        acc=evaluate(multimodel,entity_model,val_dataloader,accelerator)
        if acc>best_acc:
            best_acc=acc
        name = f'{model_name}_epoch_{epoch+1}_acc_{acc}_{round(time())}.bin'
        if accelerator.is_main_process:
            save_model(multimodel, name, accelerator)
    accelerator.print(
            '------------------------------------------------------------------------------------------')
    if accelerator.is_main_process:
        writer.close()
    if accelerator.is_main_process:
        logger.info('Training over.')