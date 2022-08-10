from dataclasses import dataclass

import torch


@dataclass
class MultiEncoderConfig:
    num_layers: int = 6
    d_text: int = 128
    d_vision: int = 14**2 + 1
    d_bottleneck: int = 4
    hidden_size: int = 768
    nhead: int = 4
    batch_first = True
    batch_size = 4 # 要和预处理的batch_size统一
    epochs = 10
