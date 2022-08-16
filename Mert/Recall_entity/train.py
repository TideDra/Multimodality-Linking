import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent)
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
from accelerate import Accelerator

accelerator = Accelerator()
accelerator.free_memory()
from apex.optimizers import FusedAdam
import logging
from MNER.utils import getlogger,train, evaluate, save_model

logger = getlogger('Mert')
if accelerator.is_main_process:
    logger.info('Loading packages...')
from MNER.model import MertForNERwithESD_bert_only
from MNER.config import config
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from transformers import get_scheduler
import warnings
from time import time
from MNER.dataset import TwitterDatasetV2, TwitterColloteFnV2, DataLoaderX,TwitterDataset,TwitterColloteFn

#silence logs
warnings.filterwarnings("ignore")
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    if log_name != "Mert" and log_name != 'accelerate':
        log_obj.disabled = True

device = config.device

