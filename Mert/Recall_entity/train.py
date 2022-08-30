import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
from accelerate import Accelerator

accelerator = Accelerator()
accelerator.free_memory()
from torch.optim import AdamW
import logging

from Datasets.EntityLink_dataset import getEntityLinkDataLoaderV3,EntityLinkDatasetConfig
dataset_config=EntityLinkDatasetConfig()
from Recall_entity.utils import trainV6,evaluateV6_2,getlogger,save_model
logger = getlogger('Mert')
if accelerator.is_main_process:
    logger.info('Loading packages...')
from multi_encoder.model import MultiEncoder,MultiEncoderConfig


from torch.utils.tensorboard import SummaryWriter
import torch
from transformers import get_scheduler,AutoModel,BertTokenizerFast
import warnings
from time import time
from Recall_entity.config import config
from Recall_entity.model import MertForEL
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

    #multimodel=MultiEncoder.from_pretrained('/root/Multimodality-Link/Mert/MultiEncoder_ckpt/me-flickr_2.pkl',MultiEncoderConfig())
    #entity_model=AutoModel.from_pretrained('bert-base-cased')
    multimodel=MertForEL()
    multimodel.load_state_dict(torch.load('/root/Multimodality-Link/Mert/Recall_entity/checkpoints/MertForEL/MertForEL_epoch_2.bin'))
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
    
    train_dataloader,val_dataloader,test_dataloader=getEntityLinkDataLoaderV3()
    if accelerator.is_main_process:
        logger.info('Done.')

    optimizer = AdamW([{'params':multimodel.parameters(), 'lr':config.learning_rate}])
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=config.epochs * len(train_dataloader),
    )
    if accelerator.is_main_process:
        logger.info('Launching training.')
    multimodel,lr_scheduler, train_dataloader, optimizer = accelerator.prepare(
        multimodel, lr_scheduler, train_dataloader, optimizer)
    #entity_model=accelerator.prepare(entity_model)
    val_dataloader=accelerator.prepare(val_dataloader)
    test_dataloader=accelerator.prepare(test_dataloader)
    criterion=torch.nn.CrossEntropyLoss()
    
    best_acc=0
    entity_processor=BertTokenizerFast.from_pretrained('bert-base-cased')
    for epoch in range(config.epochs):
        loss = trainV6(multimodel,criterion,train_dataloader, optimizer, lr_scheduler,
                     epoch + 1, writer, accelerator)
        
        if accelerator.is_main_process:
            writer.add_scalar('train/epoch_loss', loss, epoch + 1)
        accelerator.print('\n')

        acc=evaluateV6_2(multimodel,test_dataloader,accelerator)
        #name = f'{model_name}_epoch_{epoch+1}.bin'
        ##name2=f'{entity_model.__class__.__name__}_epoch_{epoch+1}.bin'
        #if accelerator.is_main_process:
        #    save_model(multimodel, name, accelerator)
        ##    save_model(entity_model,name2,accelerator)
    accelerator.print(
            '------------------------------------------------------------------------------------------')
    if accelerator.is_main_process:
        writer.close()
    if accelerator.is_main_process:
        logger.info('Training over.')