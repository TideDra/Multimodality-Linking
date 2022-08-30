import torch
from torch import nn, Tensor

from Mert.multi_encoder.model import MultiEncoder
from Mert.common_config import PretrainedModelConfig

class LinkLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(768 * 2, 768 * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.seq(input)


class ProjectLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(768, 768 * 2),
            nn.ReLU(),
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.seq(input)


class MertForEL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.multi_encoder = MultiEncoder.from_pretrained(PretrainedModelConfig.multiencoder_path)
        self.seq = nn.Sequential(nn.Linear(768, 768), nn.Tanh(), nn.Linear(768, 2))

    def forward(self, **input) -> Tensor:
        output = self.multi_encoder(**input).text_embeddings[:, 0]
        output = self.seq(output)
        return output

    @classmethod
    def from_pretrained(cls, f, *args, **kwargs):
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(f, map_location="cpu"))
        return model
