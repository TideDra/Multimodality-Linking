from os import PathLike
from pathlib import Path
from typing import List

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .model import MultiEncoderOutput
from .train_config import MultiEncoderTrainConfig


def train(
    model: MultiEncoderOutput, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler, epoch: int, config: MultiEncoderTrainConfig,
    writer: SummaryWriter, accelerator: Accelerator
):
    model.train()
    total_loss = 0.0
    total_batch = 0
    with tqdm(
        enumerate(dataloader, start=1),
        unit='batch',
        total=len(dataloader),
        desc=f'epoch:{epoch}/{config.epochs}',
        disable=not accelerator.is_local_main_process
    ) as tbar:
        for idx, batch in tbar:
            batch_inputs = batch["batch_inputs"]
            loss = model(**batch_inputs)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            loss = accelerator.gather(loss)
            total_loss += loss.sum().item()
            total_batch += batch_inputs["input_ids"].size(0)
            if accelerator.is_main_process:
                if idx % config.save_state_interval == 0:
                    accelerator.save_state(config.state_path)
                writer.add_scalar('train/batch_loss', loss.sum().item(), len(dataloader) * (epoch - 1) + idx)
            tbar.set_postfix(loss=f"{total_loss / total_batch:.6f}")
            tbar.update()
    return total_loss


def evaluate(
    model: MultiEncoderOutput, dataloader: DataLoader, config: MultiEncoderTrainConfig, accelerator: Accelerator
):
    model.eval()
    total_loss = 0.0
    total_batch = 0
    with torch.no_grad():
        with tqdm(
            enumerate(dataloader),
            unit='batch',
            total=len(dataloader),
            desc=f'Evaluating...',
            disable=not accelerator.is_local_main_process
        ) as tbar:
            for idx, batch in tbar:
                batch_inputs = batch["batch_inputs"]
                loss = model(**batch_inputs)
                loss = accelerator.gather(loss)
                total_loss += loss.sum().item()
                total_batch += batch_inputs["input_ids"].size(0)
                tbar.set_postfix(loss=f"{total_loss / total_batch:.6f}")
                tbar.update()
    return total_loss / total_batch


def get_ckpt_list(config: MultiEncoderTrainConfig) -> List[Path]:
    if not Path(config.ckpt_path).exists():
        return []
    ckpt_list = [s for s in Path(config.ckpt_path).iterdir() if s.stem[: s.stem.rindex('_')] == config.ckpt_name]
    ckpt_list.sort(key=lambda s: int(s.stem[s.stem.rindex("_") + 1 :]), reverse=True)
    return ckpt_list


def save_model(model: MultiEncoderOutput, epoch: int, config: MultiEncoderTrainConfig, accelerator: Accelerator):
    ckpt_path = Path(config.ckpt_path)
    if not ckpt_path.exists():
        ckpt_path.mkdir()

    accelerator.print('Saving checkpoint...\n')
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    # Example: model_10.pkl
    ckpt_list = get_ckpt_list(config)
    if len(ckpt_list) >= config.max_ckpt_num:
        for del_path in ckpt_list[config.max_ckpt_num - 1 :]:
            del_path.unlink()

    ckpt = {
        "config": unwrapped_model.encoder.config.to_json(),
        "model_state_dict": unwrapped_model.state_dict(),
        "epoch": epoch,
    }
    accelerator.save(ckpt, ckpt_path / f"{config.ckpt_name}_{epoch}.pkl")
    accelerator.print('Checkpoint has been updated successfully.\n')


def load_model(path: str, model: torch.nn.Module) -> dict:
    ckpt = torch.load(path,map_location='cpu')
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt["model_state_dict"]
    return ckpt


def load_model_best(config: MultiEncoderTrainConfig, model: torch.nn.Module) -> dict:
    ckpt_list = get_ckpt_list(config)
    if len(ckpt_list) == 0:
        return None
    ckpt = torch.load(ckpt_list[0])
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt["model_state_dict"]
    return ckpt
