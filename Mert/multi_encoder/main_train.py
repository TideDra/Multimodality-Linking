import logging
import sys
import warnings
from pathlib import Path

from accelerate import Accelerator

accelerator = Accelerator()
accelerator.free_memory()

sys.path.append(str(Path(__file__).resolve().parent.parent))

from multi_encoder.train import main_train
from MNER.utils import getlogger

if __name__ == "__main__":

    logger = getlogger('Mert')
    if accelerator.is_main_process:
        logger.info('Loading packages...')

    warnings.filterwarnings("ignore")

    for log_name, log_obj in logging.Logger.manager.loggerDict.items():
        if log_name != "Mert" and log_name != 'accelerate':
            log_obj.disabled = True

    main_train(accelerator, logger)
