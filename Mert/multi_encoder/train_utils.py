import torch
from accelerate import Accelerator
from Mert.multi_encoder.config import MultiEncoderConfig
from Mert.multi_encoder.model import MultiEncoderOutput
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(
    model: MultiEncoderOutput, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler, epoch: int, config: MultiEncoderConfig, writer: SummaryWriter,
    accelerator: Accelerator
):
    model.train()
    total_loss = 0.0
    with tqdm(
        enumerate(dataloader, start=1),
        unit='batch',
        total=len(dataloader),
        desc=f'epoch:{epoch}/{config.epochs}',
        disable=not accelerator.is_local_main_process
    ) as tbar:
        for idx, (inputs, _, _) in tbar:
            loss = model(**inputs)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            loss = accelerator.gather(loss)
            total_loss += loss.sum().item()
            if accelerator.is_main_process:
                writer.add_scalar('train/batch_loss', loss.sum().item(), len(dataloader) * (epoch - 1) + idx)
            tbar.set_postfix(loss=f"{(total_loss / idx) / config.batch_size:.6f}")
            tbar.update()
    return total_loss


def evaluate(model: MultiEncoderOutput, dataloader: DataLoader, config: MultiEncoderConfig, accelerator: Accelerator):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, _, _ in tqdm(
            dataloader,
            unit='batch',
            total=len(dataloader),
            desc='Evaluating...',
            disable=not accelerator.is_local_main_process
        ):
            loss = model(**inputs)
            loss = accelerator.gather(loss)
            total_loss += loss.sum().item()
    return total_loss / len(dataloader) / config.batch_size