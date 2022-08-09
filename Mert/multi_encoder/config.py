from dataclasses import dataclass

import torch


@dataclass
class MultiEncoderConfig:
    num_layers: int = 16
    d_text: int = 256
    d_vision: int = 256
    d_bottleneck: int = 4
    hidden_size: int = 768
    nhead: int = 4
    batch_first = True
    batch_size = 4 # 要和预处理的batch_size统一
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
