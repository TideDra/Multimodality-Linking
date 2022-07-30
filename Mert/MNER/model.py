from turtle import forward
from torch import nn
from torchcrf import CRF
import torch

'''
encoder要有属性hidden_size
encoder的最终输出要提供获取文本特征的接口,
注意padding

'''


class MertForNER(nn.Module):
    def __init__(self, encoder, num_tags) -> None:
        super().__init__()
        self.encoder = encoder
        self.crf = CRF(num_tags=num_tags, batch_first=True)
        self.classifier = nn.Linear(encoder.config.hidden_size, num_tags)

    def forward(self, inputs, labels):
        '''
        inputs为由Processor返回的多模态标准输入格式
        '''
        hidden_states = self.encoder(**inputs).text_embeddings
        logits = self.classifier(hidden_states)
        loss = self.crf(logits,
                        labels,
                        attention_mask=inputs['attention_mask'])
        loss = -1 * loss
        return logits, loss


class MertForNERwithESD(nn.Module):
    def __init__(self, encoder, ESD_encoder, num_tags, ESD_num_tags,
                 ratio) -> None:
        super().__init__()
        self.encoder = encoder
        self.ESD_encoder = ESD_encoder
        self.crf = CRF(num_tags=num_tags, batch_first=True)
        self.ESD_crf = CRF(num_tags=ESD_num_tags, batch_first=True)
        self.classifier = nn.Linear(encoder.config.hidden_size, num_tags)
        self.ESD_classifier = nn.Linear(encoder.config.hidden_size, ESD_num_tags)
        self.ratio = ratio

    def forward(self, inputs, labels, ESD_labels, W_e2n):
        hidden_states = self.encoder(**inputs).text_embeddings
        logits = self.classifier(hidden_states)

        del inputs['pixel_values']
        ESD_hidden_states = self.ESD_encoder(**inputs).last_hidden_state
        ESD_logits = self.ESD_classifier(ESD_hidden_states)
        ESD_loss = self.ESD_crf(ESD_logits,
                                ESD_labels,
                                attention_mask=inputs['attention_mask'])
        ESD_loss = -1 * ESD_loss

        logits += torch.bmm(ESD_logits, W_e2n)
        loss = self.crf(logits,
                        labels,
                        attention_mask=inputs['attention_mask'])
        loss = -1 * loss

        total_loss = loss + self.ratio * ESD_loss
        return logits, total_loss


