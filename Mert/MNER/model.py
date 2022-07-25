from turtle import forward
from torch import nn
from torchcrf import CRF
import torch
from utils import get_W_e2n_
'''
encoder要有属性hidden_size
encoder的最终输出要提供获取文本特征的接口,
注意padding

'''
class MertForNER(nn.Module):
    def __init__(self,encoder,num_tags) -> None:
        super().__init__()
        self.encoder=encoder
        self.crf=CRF(num_tags=num_tags,batch_first=True)
        self.classifier = nn.Linear(encoder.hidden_size,num_tags)
    
    def forward(self,text_input,img_input,labels):
        '''
        text_input为transformers库的标准文本输入格式,这个要与encoder作者保持一致,此外还需文本序列的标签
        '''
        hidden_states = self.encoder(text_input,img_input).text_features
        logits = self.classifier(hidden_states)
        loss=self.crf(logits,labels,attention_mask=text_input['attention_mask'])
        loss = -1*loss
        return logits,loss

class MertForNERwithESD(nn.Module):
    def __init__(self,encoder,ESD_encoder,num_tags,ESD_num_tags,ratio) -> None:
        super().__init__()
        self.encoder=encoder
        self.ESD_encoder=ESD_encoder
        self.crf=CRF(num_tags=num_tags,batch_first=True)
        self.ESD_crf=CRF(num_tags=ESD_num_tags,batch_first=True)
        self.classifier = nn.Linear(encoder.hidden_size,num_tags)
        self.ESD_classifier = nn.Linear(encoder.hidden_size,ESD_num_tags)
        self.ratio=ratio

    def forward(self,text_input,img_input,labels,ESD_labels,W_e2n):
        hidden_states = self.encoder(text_input,img_input).text_features
        logits = self.classifier(hidden_states)
        

        ESD_hidden_states = self.ESD_encoder(text_input)
        ESD_logits = self.ESD_classifier(ESD_hidden_states)
        ESD_loss = self.ESD_crf(ESD_logits,ESD_labels,attention_mask=text_input['attention_mask'])
        ESD_loss=-1*ESD_loss

        logits+=torch.bmm(ESD_logits,W_e2n)
        loss=self.crf(logits,labels,attention_mask=text_input['attention_mask'])
        loss = -1*loss

        total_loss=loss+self.ratio*ESD_loss
        return logits,total_loss


