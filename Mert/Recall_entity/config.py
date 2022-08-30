from dataclasses import dataclass
import torch
@dataclass
class Config:
    epochs:int=30
    learning_rate=5*1e-5
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tb_dir='/root/Multimodality-Link/Mert/Recall_entity/tb_log'
    checkpoint_path='/root/Multimodality-Link/Mert/Recall_entity/checkpoints'
    max_checkpoint_num=2
config=Config()