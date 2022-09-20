from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
from torch import Tensor, nn
from transformers import FlavaModel, FlavaTextModel, FlavaImageModel, FlavaMultimodalModel
from transformers.models.flava.modeling_flava import FlavaModelOutput, BaseModelOutputWithPooling

from .config import MultiEncoderConfig
from .transformer import TransformerEncoderLayer2


@dataclass
class PairOutput:
    text_embeddings: Tensor
    image_embeddings: Tensor
    prev: "PairOutput" = None


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
        src_text, bottleneck = torch.split(y, [src_text.size(cat_dim), bottleneck.size(cat_dim)], dim=cat_dim)
        z = self.trans_vb(torch.cat([src_vision, bottleneck], dim=cat_dim))
        src_vision, bottleneck = torch.split(z, [src_vision.size(cat_dim), bottleneck.size(cat_dim)], dim=cat_dim)
        return FusionModelOutput(src_text, src_vision, bottleneck)


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


class GlobalFusionModel(nn.Module):
    def __init__(self, config: MultiEncoderConfig):
        super().__init__()
        self.config = config
        # encoder_layer = nn.TransformerEncoderLayer(config.hidden_size, config.nhead)
        # self.trans1 = nn.TransformerEncoder(encoder_layer, config.num_layers)
        # self.trans2 = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.trans1 = FlavaMultimodalModel.from_pretrained('facebook/flava-full', use_cls_token=False)
        self.trans2 = FlavaMultimodalModel.from_pretrained('facebook/flava-full', use_cls_token=False)

    def forward(self, token_embeddings: Tensor, image_embeddings: Tensor, phrase_embeddings: Tensor) -> PairOutput:
        cat_dim = 1
        if phrase_embeddings is not None:
            x1 = self.trans1(torch.cat((token_embeddings, image_embeddings), dim=cat_dim)).last_hidden_state
            x2 = self.trans2(torch.cat((phrase_embeddings, image_embeddings), dim=cat_dim)).last_hidden_state
            x11, x12 = torch.split(x1, [token_embeddings.size(cat_dim), image_embeddings.size(cat_dim)], dim=cat_dim)
            x21, x22 = torch.split(x2, [phrase_embeddings.size(cat_dim), image_embeddings.size(cat_dim)], dim=cat_dim)
            y1 = torch.cat((x11, x21), dim=cat_dim)
            y2 = torch.cat((x12, x22), dim=cat_dim)
        else:
            x1 = self.trans1(torch.cat((token_embeddings, image_embeddings), dim=cat_dim)).last_hidden_state
            y1, y2 = torch.split(x1, [token_embeddings.size(cat_dim), image_embeddings.size(cat_dim)], dim=cat_dim)
        return PairOutput(text_embeddings=y1, image_embeddings=y2)


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


class PhraseLevelExtractorV2(nn.Module):
    '''Extract phrase level information with multiple conv kernels.'''
    def __init__(self, hidden_size: int, kernel_sizes: List[int]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2)
            for kernel_size in kernel_sizes
        ])
        self.relu = nn.ReLU()

    def forward(self, text_embeddings: Tensor) -> Tensor:
        x = text_embeddings.permute(0, 2, 1)
        y = torch.zeros_like(x) - 1e6
        for conv in self.convs:
            y = torch.maximum(conv(x), y)
        y = y.permute(0, 2, 1)
        y = self.relu(y)
        return y


class MultiEncoderBase(nn.Module):
    '''Base MultiEncoder'''
    def __init__(self, config: MultiEncoderConfig):
        super().__init__()
        self.config = config

    def pass_flava(self, **batch_data) -> PairOutput:
        image_outputs: BaseModelOutputWithPooling = self.flava_image(pixel_values=batch_data["pixel_values"])
        text_outputs: BaseModelOutputWithPooling = self.flava_text(
            input_ids=batch_data["input_ids"],
            token_type_ids=batch_data["token_type_ids"],
            attention_mask=batch_data["attention_mask"]
        )
        return PairOutput(text_outputs.last_hidden_state, image_outputs.last_hidden_state)

    def pass_phrase(self, text_embeddings: Tensor) -> Tensor:
        return self.phrase_level(text_embeddings)

    def pass_multimodal(self, pair: PairOutput, return_dict=True) -> Union[PairOutput, Tuple[Tensor, Tensor]]:
        flava_output: BaseModelOutputWithPooling = self.flava_multimodal(
            torch.cat([pair.text_embeddings, pair.image_embeddings], dim=1)
        )
        text_embeddings, image_embeddings = torch.split(
            flava_output.last_hidden_state,
            [pair.text_embeddings.size(1), pair.image_embeddings.size(1)], dim=1
        )
        if return_dict:
            return PairOutput(text_embeddings, image_embeddings, prev=pair if self.config.forward_link else None)
        else:
            return text_embeddings, image_embeddings

    def pass_globalfusion(self,
                          pair: PairOutput,
                          phrase: Tensor,
                          return_dict=True) -> Union[PairOutput, Tuple[Tensor, Tensor]]:
        output: PairOutput = self.global_fusion(
            token_embeddings=pair.text_embeddings, image_embeddings=pair.image_embeddings, phrase_embeddings=phrase
        )
        if return_dict:
            output.prev = pair if self.config.forward_link else None
            return output
        else:
            return output.text_embeddings, output.image_embeddings

    def pass_bottleneck(self, pair: PairOutput, return_dict=True) -> Union[PairOutput, Tuple[Tensor, Tensor]]:
        fusion_output: FusionModelOutput = self.fusion(pair.text_embeddings, pair.image_embeddings)
        text_embeddings, image_embeddings = fusion_output.text_embeddings, fusion_output.image_embeddings
        if return_dict:
            return PairOutput(text_embeddings, image_embeddings, prev=pair if self.config.forward_link else None)
        else:
            return text_embeddings, image_embeddings

    @classmethod
    def from_pretrained(cls, path: str, **config_args):
        '''
        Args:
            param path (str): Path of checkpoint of pretrained model.  
            config_args (optional): Supplement of MultiEncoderConfig arguments.
        Returns:
            Pretrained model.
        '''
        ckpt = torch.load(path, map_location="cpu")
        if "config" in ckpt:
            config = MultiEncoderConfig.from_json(ckpt["config"])
        else:
            config = MultiEncoderConfig()
        for k, v in config_args.items():
            setattr(config, k, v)
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
        print("Init MultiEncoder", config.to_json())
        print(f"PhraseLevel: {config.augment_text}")
        print(f"GlobalFusion: {'gf' in config.passes}, BottleneckFusion: {'bn' in config.passes}")
        if config.sole_flava:
            self.flava = FlavaModel.from_pretrained('facebook/flava-full')
        else:
            self.flava_text = FlavaTextModel.from_pretrained('facebook/flava-full')
            self.flava_image = FlavaImageModel.from_pretrained('facebook/flava-full')
        if "mm" in config.passes:
            self.flava_multimodal = FlavaMultimodalModel.from_pretrained('facebook/flava-full', use_cls_token=False)
        else:
            self.global_fusion = GlobalFusionModel(config)
        self.fusion = MultiFusionModel(config)
        if config.augment_text:
            self.phrase_level = PhraseLevelExtractor(config.hidden_size) if "mm" in config.passes \
                else PhraseLevelExtractorV2(config.hidden_size, config.conv_kernel_sizes)

    def forward(self, **batch_data) -> PairOutput:
        if self.config.sole_flava:
            flava_output: FlavaModelOutput = self.flava(**batch_data)
            output = PairOutput(flava_output.text_embeddings, flava_output.image_embeddings)
        else:
            output = self.pass_flava(**batch_data)
        phrase = None
        if self.config.augment_text:
            phrase = self.phrase_level(output.text_embeddings)
        for p in self.config.passes:
            if p == "mm":
                output = self.pass_multimodal(output)
            elif p == "gf":
                output = self.pass_globalfusion(output, phrase)
            elif p == "bn":
                output = self.pass_bottleneck(output)
            else:
                raise
        return output


class MultiEncoderOutput(nn.Module):
    '''Used for training MultiEncoder with contrastive loss. Adapted from CLIP.'''
    def __init__(self, encoder: MultiEncoderBase):
        super().__init__()
        self.encoder = encoder
        self.logit_scale = nn.Parameter(torch.log(torch.tensor([1 / 0.07])))
        self.crit_i = nn.CrossEntropyLoss()
        self.crit_t = nn.CrossEntropyLoss()

    def forward(self, **batch_data) -> Tensor:
        outputs: PairOutput = self.encoder(**batch_data)
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
