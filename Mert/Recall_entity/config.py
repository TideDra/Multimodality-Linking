from dataclasses import dataclass
import torch
@dataclass
class Config:
    batch_size:int=8
    epochs:int=10
    learning_rate=5*1e-5
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tb_dir='/home/zero_lag/Document/srtp/Multimodality-Link/Mert/Recall_entity/tb_log'
    checkpoint_path='/home/zero_lag/Document/srtp/Multimodality-Link/Mert/Recall_entity/checkpoints'
config=Config()