from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import FlavaModel, FlavaTextModel, FlavaImageModel, FlavaMultimodalModel, FlavaFeatureExtractor
from transformers.models.flava.modeling_flava import FlavaModelOutput, BaseModelOutputWithPooling

from .config import MultiEncoderConfig
from .transformer import TransformerEncoderLayer2


@dataclass
class FusionModelOutput:
    text_embeddings: Tensor
    image_embeddings: Tensor
    bottleneck: Tensor = None


class MultiFusionLayer(nn.Module):
    '''
    Direct bottleneck fusion from paper. In each layer, applay transformer to text-bottleneck first and then image-bottlenect.
    Outputs final text, image, bottlenect embeddings.
    '''
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


class MultiFusionLayerV2(nn.Module):
    '''
    Use custom TransformerEncoderLayer to transform text, bottle, and image simultaneously.\\
    Has faster loss decline but performs worse in downstream tasks.
    '''
    def __init__(self, config: MultiEncoderConfig):
        super().__init__()
        self.config = config

        d_model = config.hidden_size
        nhead = config.nhead

        self.trans_tb = TransformerEncoderLayer2(
            d_model,
            nhead,
            config.d_text + config.d_bottleneck,
            config.d_text,
            batch_first=config.batch_first,
        )
        self.trans_vb = TransformerEncoderLayer2(
            d_model,
            nhead,
            config.d_vision + config.d_bottleneck,
            config.d_vision,
            batch_first=config.batch_first,
        )
        self.trans_tvb = TransformerEncoderLayer2(
            d_model,
            nhead,
            config.d_text + config.d_vision + config.d_bottleneck,
            config.d_bottleneck,
            batch_first=config.batch_first
        )

    def forward(self, src_text: Tensor, src_vision: Tensor, bottleneck: Tensor) -> FusionModelOutput:
        # shape: batch_size * len * d_model
        cat_dim = 1
        out_text = self.trans_tb(torch.cat([src_text, bottleneck], dim=cat_dim), src_text)
        out_vision = self.trans_vb(torch.cat([src_vision, bottleneck], dim=cat_dim), src_vision)
        out_btn = self.trans_tvb(torch.cat([src_text, src_vision, bottleneck], dim=cat_dim), bottleneck)
        return FusionModelOutput(out_text, out_vision, out_btn)


class MultiFusionModel(nn.Module):
    '''Container of fusion layers.'''
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


class MultiEncoderBase(nn.Module):
    '''Base MultiEncoder'''
    def __init__(self, config: MultiEncoderConfig):
        super().__init__()
        self.config = config

    @classmethod
    def from_pretrained(cls, path: str, config: MultiEncoderConfig = None):
        ckpt = torch.load(path)
        if "config" in ckpt:
            config = ckpt["config"]
        encoder = cls(config)
        model = MultiEncoderOutput(encoder=encoder)
        model.load_state_dict(ckpt["model_state_dict"])
        return model.encoder


class MultiEncoder(MultiEncoderBase):
    '''
    Core of multimodal feature extraction and fusion. Outputs fusion embeddings.\\
    Use output embeddings of text and image from FlavaModel, and then directly do fusion work.
    '''
    def __init__(self, config=MultiEncoderConfig()):
        super().__init__(config)
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


class MultiEncoderV1(MultiEncoderBase):
    '''
    Add extra FlavaMultimodalModel before FusionModel.
    '''
    def __init__(self, config=MultiEncoderConfig()):
        super().__init__(config)
        self.flava_text = FlavaTextModel.from_pretrained('facebook/flava-full')
        self.flava_image = FlavaImageModel.from_pretrained('facebook/flava-full')
        self.flava_multimodal = FlavaMultimodalModel.from_pretrained('facebook/flava-full', use_cls_token=False)
        self.fusion = MultiFusionModel(config)

    def forward(self, **batch_data) -> FusionModelOutput:
        image_outputs: BaseModelOutputWithPooling = self.flava_image(pixel_values=batch_data["pixel_values"])
        text_outputs: BaseModelOutputWithPooling = self.flava_text(
            input_ids=batch_data["input_ids"],
            token_type_ids=batch_data["token_type_ids"],
            attention_mask=batch_data["attention_mask"]
        )
        image_embeddings = image_outputs.last_hidden_state
        text_embeddings = text_outputs.last_hidden_state

        flava_output: BaseModelOutputWithPooling = self.flava_multimodal(
            torch.cat([text_embeddings, image_embeddings], dim=1)
        )
        fusion_output: FusionModelOutput = self.fusion(
            *torch.split(flava_output.last_hidden_state,
                         [text_embeddings.size(1), image_embeddings.size(1)], dim=1)
        )
        return fusion_output


class PhraseLevelExtractor(nn.Module):
    '''Extract phrase level information with 3 conv kernels.'''
    def __init__(self, hidden_size: int):
        super().__init__()
        self.conv_1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv_2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=2, padding=1, dilation=2)
        self.conv_3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)

    def forward(self, text_embeddings: Tensor) -> Tensor:
        x = text_embeddings.permute(0, 2, 1)
        x = torch.maximum(torch.maximum(self.conv_1(x), self.conv_2(x)), self.conv_3(x))
        return x.permute(0, 2, 1)


class MultiEncoderV2_1(MultiEncoderBase):
    '''
    Add extra Conv1D to text embeddings process.\\
    Also add extra FlavaMultimodalModel to the end of FusionModel.
    '''
    def __init__(self, config=MultiEncoderConfig()):
        super().__init__(config)
        self.flava_text = FlavaTextModel.from_pretrained('facebook/flava-full')
        self.flava_image = FlavaImageModel.from_pretrained('facebook/flava-full')
        self.flava_multimodal = FlavaMultimodalModel.from_pretrained('facebook/flava-full', use_cls_token=False)
        self.fusion = MultiFusionModel(config)
        self.phrase_level = PhraseLevelExtractor(config.hidden_size)

    def forward(self, **batch_data) -> FusionModelOutput:
        image_outputs: BaseModelOutputWithPooling = self.flava_image(pixel_values=batch_data["pixel_values"])
        text_outputs: BaseModelOutputWithPooling = self.flava_text(
            input_ids=batch_data["input_ids"],
            token_type_ids=batch_data["token_type_ids"],
            attention_mask=batch_data["attention_mask"]
        )
        image_embeddings = image_outputs.last_hidden_state
        text_embeddings = text_outputs.last_hidden_state
        text_embeddings = self.phrase_level(text_embeddings)
        fusion_output: FusionModelOutput = self.fusion(text_embeddings, image_embeddings)
        flava_output: BaseModelOutputWithPooling = self.flava_multimodal(
            torch.cat([fusion_output.text_embeddings, fusion_output.image_embeddings], dim=1)
        )
        return FusionModelOutput(
            *torch.split(flava_output.last_hidden_state,
                         [text_embeddings.size(1), image_embeddings.size(1)], dim=1)
        )


class MultiEncoderV2_2(MultiEncoderBase):
    '''
    Add extra Conv1D to text embeddings process.\\
    Also add extra FlavaMultimodalModel before FusionModel.
    '''
    def __init__(self, config=MultiEncoderConfig()):
        super().__init__(config)
        self.flava_text = FlavaTextModel.from_pretrained('facebook/flava-full')
        self.flava_image = FlavaImageModel.from_pretrained('facebook/flava-full')
        self.flava_multimodal = FlavaMultimodalModel.from_pretrained('facebook/flava-full', use_cls_token=False)
        self.fusion = MultiFusionModel(config)
        self.phrase_level = PhraseLevelExtractor(config.hidden_size)

    def forward(self, **batch_data) -> FusionModelOutput:
        image_outputs: BaseModelOutputWithPooling = self.flava_image(pixel_values=batch_data["pixel_values"])
        text_outputs: BaseModelOutputWithPooling = self.flava_text(
            input_ids=batch_data["input_ids"],
            token_type_ids=batch_data["token_type_ids"],
            attention_mask=batch_data["attention_mask"]
        )
        '''
        last_hidden_state: torch.FloatTensor = None
        image torch.Size([20, 197, 768]) torch.Size([20, 768])
        text torch.Size([20, 64, 768]) torch.Size([20, 768])
        '''
        image_embeddings = image_outputs.last_hidden_state
        text_embeddings = text_outputs.last_hidden_state
        text_embeddings = self.phrase_level(text_embeddings)

        flava_output: BaseModelOutputWithPooling = self.flava_multimodal(
            torch.cat([text_embeddings, image_embeddings], dim=1)
        )
        fusion_output: FusionModelOutput = self.fusion(
            *torch.split(flava_output.last_hidden_state,
                         [text_embeddings.size(1), image_embeddings.size(1)], dim=1)
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
