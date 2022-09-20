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
    def __init__(self, mert_config: dict = None) -> None:
        super().__init__()
        if not mert_config:
            mert_config = {}
        self.multi_encoder = MultiEncoder.from_pretrained(PretrainedModelConfig.multiencoder_path, **mert_config)
        self.seq = nn.Sequential(nn.Linear(768, 768), nn.Tanh(), nn.Linear(768, 2))

    def forward(self, **input) -> Tensor:
        output = self.multi_encoder(**input).text_embeddings[:, 0]
        output = self.seq(output)
        return output

    @classmethod
    def from_pretrained(cls, f, *args, **kwargs):
        model = cls(*args, **kwargs)
        state_dict = torch.load(f, map_location="cpu")
        if "fc.weight" in state_dict:
            state_dict2 = {
                "0.weight": state_dict["fc.weight"],
                "0.bias": state_dict["fc.bias"],
                "2.weight": state_dict["classifier.weight"],
                "2.bias": state_dict["classifier.bias"],
            }
            model.seq.load_state_dict(state_dict2)
        else:
            model.load_state_dict(state_dict)
        return model
