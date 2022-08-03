from dataclasses import dataclass


@dataclass
class MultimodalFusionConfig:
    num_layers: int = 16
    d_text: int = 256
    d_vision: int = 256
    d_bottleneck: int = 4
    hidden_size: int = 768
    nhead: int = 8
