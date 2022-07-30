from torch import nn
from torchcrf import CRF
import torch
from transformers import FlavaModel, BertModel
from config import config
'''
encoder要有属性hidden_size
encoder的最终输出要提供获取文本特征的接口,
注意padding

'''


class ModelForNER(nn.Module):
    def __init__(self, encoder, num_tags, is_encoder_frozen=True,dropout=0.1) -> None:
        super().__init__()
        self.encoder = encoder
        self.crf = CRF(num_tags=num_tags, batch_first=True)
        self.classifier = nn.Linear(encoder.config.hidden_size, num_tags)
        self.dropout=nn.Dropout(dropout)
        self.is_encoder_frozen=is_encoder_frozen
    def forward(self, inputs, labels):
        '''
        inputs为由Processor返回的多模态标准输入格式
        '''
        if self.is_encoder_frozen:
            with torch.no_grad():
                hidden_states = self.encoder(**inputs).text_embeddings
        else:
            hidden_states = self.encoder(**inputs).text_embeddings
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        loss = self.crf(logits,
                        labels,
                        mask=inputs['attention_mask'])
        loss = -1 * loss
        return logits, loss


class ModelForNERwithESD(nn.Module):
    def __init__(self, encoder, ESD_encoder, num_tags, ESD_num_tags,
                 ratio, is_encoder_frozen=True, is_ESD_encoder_frozen=True,dropout=0.1) -> None:
        super().__init__()
        self.encoder = encoder
        self.ESD_encoder = ESD_encoder
        self.crf = CRF(num_tags=num_tags, batch_first=True)
        self.ESD_crf = CRF(num_tags=ESD_num_tags, batch_first=True)
        self.classifier = nn.Linear(encoder.config.hidden_size, num_tags)
        self.ESD_classifier = nn.Linear(encoder.config.hidden_size, ESD_num_tags)
        self.ratio = ratio
        self.is_encoder_frozen=is_encoder_frozen
        self.is_ESD_encoder_frozen=is_ESD_encoder_frozen
        self.dropout=nn.Dropout(dropout)

    def forward(self, inputs, labels, ESD_labels, W_e2n):
        if self.is_encoder_frozen:
            with torch.no_grad():
                hidden_states = self.encoder(**inputs).text_embeddings
        else:
            hidden_states = self.encoder(**inputs).text_embeddings
        
        hidden_states=self.dropout(hidden_states)

        logits = self.classifier(hidden_states)

        del inputs['pixel_values']

        if self.is_ESD_encoder_frozen:
            with torch.no_grad():
                ESD_hidden_states = self.ESD_encoder(**inputs).last_hidden_state
        else:
            ESD_hidden_states = self.ESD_encoder(**inputs).last_hidden_state
        
        ESD_hidden_states=self.dropout(ESD_hidden_states)
        ESD_logits = self.ESD_classifier(ESD_hidden_states)
        ESD_loss = self.ESD_crf(ESD_logits,
                                ESD_labels,
                                mask=inputs['attention_mask'])
        ESD_loss = -1 * ESD_loss

        logits += torch.bmm(ESD_logits, W_e2n)
        loss = self.crf(logits,
                        labels,
                        mask=inputs['attention_mask'])
        loss = -1 * loss

        total_loss = loss + self.ratio * ESD_loss
        return logits, total_loss

class FlavaForNER(ModelForNER):
    def __init__(self, is_encoder_frozen=True,dropout=0.1) -> None:
        encoder=FlavaModel.from_pretrained("facebook/flava-full")
        super().__init__(encoder=encoder,num_tags=len(config.tag2id),is_encoder_frozen=is_encoder_frozen,dropout=dropout)

class FlavaForNERwithESD_bert_only(ModelForNERwithESD):
    def __init__(self, ratio=1, is_encoder_frozen=True, is_ESD_encoder_frozen=True,dropout=0.1) -> None:
        encoder=FlavaModel.from_pretrained("facebook/flava-full")
        ESD_encoder=BertModel.from_pretrained('bert-base-cased')
        num_tags=len(config.tag2id)
        ESD_num_tags=len(config.ESD_id2tag)
        super().__init__(encoder, ESD_encoder, num_tags, ESD_num_tags, ratio, is_encoder_frozen, is_ESD_encoder_frozen,dropout)
