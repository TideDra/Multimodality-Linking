import torch
from accelerate import Accelerator
from Mert.multi_encoder.config import MultiEncoderConfig
from Mert.multi_encoder.model import MultiEncoderOutput
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .train_config import MultiEncoderTrainConfig


def train(
    model: MultiEncoderOutput, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler, epoch: int, config: MultiEncoderTrainConfig,
    writer: SummaryWriter, accelerator: Accelerator
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
        for idx, batch in tbar:
            loss = model(**batch["batch_inputs"])
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


def evaluate(
    model: MultiEncoderOutput, dataloader: DataLoader, config: MultiEncoderTrainConfig, accelerator: Accelerator
):
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


from pathlib import Path


def save_model(
    model: torch.nn.Module, name: str, epoch: int, config: MultiEncoderTrainConfig, accelerator: Accelerator
):
    accelerator.print('Saving checkpoint...\n')
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    # Example: model_10.pkl
    ckpt_path = Path(config.ckpt_path)
    ckpt_list = list(ckpt_path.iterdir())
    if len(ckpt_list) >= config.max_ckpt_num:
        ckpt_list.sort(key=lambda s: int(s.stem[s.stem.index("_") + 1 :]), reverse=True)
        for del_path in ckpt_list[config.max_ckpt_num - 1 :]:
            del_path.unlink()

    ckpt = {
        "model_state_dict": unwrapped_model.state_dict(),
        "epoch": epoch,
    }
    accelerator.save(ckpt, ckpt_path / f"{name}_{epoch}.pkl")
    accelerator.print('Checkpoint has been updated successfully.\n')
