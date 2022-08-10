from dataclasses import dataclass

import torch
from transformers import FlavaModel
from transformers.models.flava.modeling_flava import FlavaModelOutput
from Mert.multi_encoder.config import MultiEncoderConfig
from torch import Tensor, nn


@dataclass
class FusionModelOutput:
    text_embeddings: Tensor
    image_embeddings: Tensor
    bottleneck: Tensor


class MultiFusionLayer(nn.Module):
    def __init__(self, config: MultiEncoderConfig):
        super().__init__()
        self.config = config
        self.trans_tb = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, nhead=config.nhead, batch_first=config.batch_first
        )
        self.trans_vb = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, nhead=config.nhead, batch_first=config.batch_first
        )

    def forward(self, src_text: Tensor, src_vision: Tensor, bottleneck: Tensor) -> FusionModelOutput:
        # shape: batch_size * len * d_model
        cat_dim = 1
        y = self.trans_tb(torch.cat([src_text, bottleneck], dim=cat_dim))
        src_text, bottleneck = torch.split(y, src_text.size(cat_dim), dim=cat_dim)
        z = self.trans_vb(torch.cat([src_vision, bottleneck], dim=cat_dim))
        src_vision, bottleneck = torch.split(z, src_vision.size(cat_dim), dim=cat_dim)
        return FusionModelOutput(src_text, src_vision, bottleneck)


class MultiFusionModel(nn.Module):
    def __init__(self, config: MultiEncoderConfig):
        super().__init__()
        self.config = config
        self.fusion_layers = nn.ModuleList([MultiFusionLayer(config) for _ in range(config.num_layers)])
        self.bottleneck = nn.Parameter(torch.zeros(config.d_bottleneck, config.hidden_size))

    def forward(self, text_embeddings: Tensor, image_embeddings: Tensor) -> FusionModelOutput:
        # Ensure batch first
        if not self.config.batch_first:
            text_embeddings = text_embeddings.permute(1, 0, 2)
            image_embeddings = image_embeddings.permute(1, 0, 2)
        batch_size = text_embeddings.size(0)
        outputs = FusionModelOutput(text_embeddings, image_embeddings, self.bottleneck.repeat(batch_size, 1, 1))
        for layer in self.fusion_layers:
            outputs = layer(outputs.text_embeddings, outputs.image_embeddings, outputs.bottleneck)
        return outputs


class MultiEncoder(nn.Module):
    '''Core of multimodal feature extraction and fusion. Outputs fusion embeddings.'''
    def __init__(self, config=MultiEncoderConfig()):
        super().__init__()
        self.config = config
        self.flava = FlavaModel.from_pretrained('facebook/flava-full')
        self.fusion = MultiFusionModel(config)

    def forward(self, **batch_data) -> FusionModelOutput:
        flava_output: FlavaModelOutput = self.flava(**batch_data)
        # print(flava_output.text_embeddings.shape, flava_output.image_embeddings.shape)
        # torch.Size([6, 128, 768]) torch.Size([6, 197, 768])
        fusion_output = self.fusion(
            text_embeddings=flava_output.text_embeddings,
            image_embeddings=flava_output.image_embeddings,
        )
        return fusion_output


class MultiEncoderOutput(nn.Module):
    '''Used for training MultiEncoder with contrastive loss.'''
    def __init__(self, encoder: MultiEncoder):
        super().__init__()
        self.encoder = encoder
        self.logit_scale = nn.Parameter(torch.log(torch.tensor([1 / 0.07])))
        self.crit_i = nn.CrossEntropyLoss()
        self.crit_t = nn.CrossEntropyLoss()

    def forward(self, **batch_data) -> Tensor:
        outputs: FusionModelOutput = self.encoder(**batch_data)
        # default batch first
        image_features = outputs.image_embeddings[:, 0, :]
        text_features = outputs.text_embeddings[:, 0, :]
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = image_features @ text_features.T * logit_scale
        logits_per_text = logits_per_image.T
        # shape = [global_batch_size, global_batch_size]
        labels = torch.arange(logits_per_image.size(0)).to(logits_per_image.device)  # labels of diagonal
        loss_i = self.crit_i(logits_per_image, labels)
        loss_t = self.crit_t(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        return loss
