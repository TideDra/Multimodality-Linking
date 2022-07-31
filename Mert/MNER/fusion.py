from dataclasses import dataclass
import torch
from torch import nn
from torch import Tensor


@dataclass
class FlavaFusionConfig:
    num_layers: int = 16
    d_text: int = 256
    d_vision: int = 256
    d_bottleneck: int = 4
    hidden_size: int = 768
    nhead: int = 8


@dataclass
class FusionModelOutput:
    text_embeddings: Tensor
    image_embeddings: Tensor
    bottleneck: Tensor


class FlavaFusionModel(nn.Module):
    def __init__(self, config: FlavaFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_layers = nn.ModuleList([FlavaFusionLayer(config) for _ in range(config.num_layers)])

    def forward(self, text_embeddings: Tensor, image_embeddings: Tensor) -> FusionModelOutput:
        batch_size = text_embeddings.shape[1]
        bottleneck = torch.zeros((self.config.d_bottleneck, batch_size, self.config.hidden_size))
        outputs = FusionModelOutput(text_embeddings, image_embeddings, bottleneck)
        for layer in self.fusion_layers:
            outputs = layer(outputs.text_embeddings, outputs.image_embeddings, outputs.bottleneck)
        return outputs


class FlavaFusionLayer(nn.Module):
    def __init__(self, config: FlavaFusionConfig):
        super().__init__()
        self.trans_tb = nn.TransformerEncoderLayer(d_model=config.d_text + config.d_bottleneck, nhead=config.nhead)
        self.trans_vb = nn.TransformerEncoderLayer(d_model=config.d_vision + config.d_bottleneck, nhead=config.nhead)

    def forward(self, src_text: Tensor, src_vision: Tensor, bottleneck: Tensor) -> FusionModelOutput:
        # shape: len * batch_size * d_model
        y = self.trans_tb(torch.cat(src_text, bottleneck, dim=0))
        src_text, bottleneck = torch.split(y, bottleneck.shape[0], dim=0)
        z = self.trans_vb(torch.cat(src_vision, bottleneck, dim=0))
        src_vision, bottleneck = torch.split(z, bottleneck.shape[0], dim=0)
        return FusionModelOutput(src_text, src_vision, bottleneck)
