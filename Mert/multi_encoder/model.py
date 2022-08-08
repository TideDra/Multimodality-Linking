from dataclasses import dataclass

import torch
from transformers import FlavaConfig, FlavaModel
from transformers.models.flava.modeling_flava import FlavaModelOutput
from Mert.multi_encoder.config import MultiEncoderConfig
from torch import Tensor, nn


@dataclass
class FusionModelOutput:
    text_embeddings: Tensor
    image_embeddings: Tensor
    bottleneck: Tensor


class MultimodalFusionLayer(nn.Module):
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
        # shape: len * batch_size * d_model or batch_size * len * d_model
        cat_dim = 1 if self.config.batch_first else 0
        y = self.trans_tb(torch.cat([src_text, bottleneck], dim=cat_dim))
        src_text, bottleneck = torch.split(y, src_text.size(cat_dim), dim=cat_dim)
        z = self.trans_vb(torch.cat([src_vision, bottleneck], dim=cat_dim))
        src_vision, bottleneck = torch.split(z, src_vision.size(cat_dim), dim=cat_dim)
        return FusionModelOutput(src_text, src_vision, bottleneck)


class MultimodalFusionModel(nn.Module):
    def __init__(self, config: MultiEncoderConfig):
        super().__init__()
        self.config = config
        self.fusion_layers = nn.ModuleList([MultimodalFusionLayer(config) for _ in range(config.num_layers)])

    def forward(self, text_embeddings: Tensor, image_embeddings: Tensor) -> FusionModelOutput:
        batch_size = text_embeddings.size(0 if self.config.batch_first else 1)
        btn_shape = (batch_size, self.config.d_bottleneck, self.config.hidden_size
                     ) if self.config.batch_first else (self.config.d_bottleneck, batch_size, self.config.hidden_size)
        bottleneck = torch.zeros(btn_shape).to(self.config.device)
        outputs = FusionModelOutput(text_embeddings, image_embeddings, bottleneck)
        for layer in self.fusion_layers:
            outputs = layer(outputs.text_embeddings, outputs.image_embeddings, outputs.bottleneck)
        return outputs


class MultiEncoder(nn.Module):
    def __init__(
        self,
        flava_config: FlavaConfig = FlavaConfig(),
        fusion_config: MultiEncoderConfig = MultiEncoderConfig(),
    ):
        super().__init__()
        self.flava = FlavaModel(flava_config)
        self.fusion = MultimodalFusionModel(fusion_config)

    def forward(self, batch_data) -> FusionModelOutput:
        flava_output: FlavaModelOutput = self.flava(**batch_data)
        # print(flava_output.text_embeddings.shape, flava_output.image_embeddings.shape)
        # torch.Size([6, 128, 768]) torch.Size([6, 197, 768])
        fusion_output = self.fusion(
            text_embeddings=flava_output.text_embeddings,
            image_embeddings=flava_output.image_embeddings,
        )
        return fusion_output


class MultiEncoderOutput(nn.Module):
    def __init__(self, encoder: MultiEncoder):
        super().__init__()
        self.encoder = encoder
        #self.image_projection = nn.Parameter(torch.empty(config.d_vision, config.hidden_size))
        #self.text_projection = nn.Parameter(torch.empty(config.d_text, config.hidden_size))
        self.logit_scale = nn.Parameter(torch.log(torch.tensor([1 / 0.07])))
        self.crit_i = nn.CrossEntropyLoss()
        self.crit_t = nn.CrossEntropyLoss()

    def forward(self, batch_data) -> Tensor:
        outputs: FusionModelOutput = self.encoder(batch_data)
        # default batch first
        image_features = outputs.image_embeddings[:, 0, :]
        text_features = outputs.text_embeddings[:, 0, :]
        #image_features = image_embeddings[:, 0, :] @ self.image_projection  #[n, d_i]
        #text_features = text_embeddings[:, 0, :] @ self.text_projection  #[n, d_t]
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = image_features @ text_features.T * logit_scale
        logits_per_text = logits_per_image.T
        # shape = [global_batch_size, global_batch_size]
        labels = torch.arange(logits_per_image.size(0))  # 对角线元素的labels
        loss_i = self.crit_i(logits_per_image, labels)
        loss_t = self.crit_t(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        return loss
