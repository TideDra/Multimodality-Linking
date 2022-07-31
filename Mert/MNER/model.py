import torch
from torch import nn
from torchcrf import CRF
import torch
from transformers import FlavaModel, FlavaTextModel
from config import config

class ModelForTokenClassification(nn.Module):
    '''
    This is the base framework for token classification with CRF.
    Need a given encoder, and other arguments.
    '''
    def __init__(self,
                 encoder,
                 num_tags:int,
                 is_encoder_frozen:bool=True,
                 dropout:int=0.1) -> None:
        super().__init__()
        self.encoder = encoder
        self.crf = CRF(num_tags=num_tags, batch_first=True)
        self.classifier = nn.Linear(encoder.config.hidden_size, num_tags)
        self.dropout = nn.Dropout(dropout)
        self.is_encoder_frozen = is_encoder_frozen

    def forward(self, inputs, labels=None):
        '''
        inputs is the outputs of Processor
        '''
        if self.is_encoder_frozen:
            with torch.no_grad():
                # for multimodel encoder, use text_embeddings to get hidden_states
                encoder_output = self.encoder(**inputs)
                if hasattr(encoder_output, 'text_embeddings'):
                    hidden_states = encoder_output.text_embeddings
                else:
                    hidden_states = encoder_output.last_hidden_state
        else:
            encoder_output = self.encoder(**inputs)
            if hasattr(encoder_output, 'text_embeddings'):
                hidden_states = encoder_output.text_embeddings
            else:
                hidden_states = encoder_output.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        if labels!=None:
            loss = self.crf(logits, labels, mask=inputs['attention_mask'].bool())
            loss = -1 * loss
        else:
            loss = None
        return logits, loss


class ModelForNERwithESD(nn.Module):
    '''
    This is the base framework for NER with ESD.
    Need a given encoder(multimodal or text model only), a given ESD_encoder(text model only), and other arguments.
    '''
    def __init__(self,
                 encoder,
                 ESD_encoder,
                 num_tags:int,
                 ESD_num_tags:int,
                 ratio:float,
                 is_encoder_frozen:bool=True,
                 is_ESD_encoder_frozen:bool=True,
                 dropout:float=0.1) -> None:
        super().__init__()
        self.NERmodel = ModelForTokenClassification(
            encoder=encoder,
            num_tags=num_tags,
            is_encoder_frozen=is_encoder_frozen,
            dropout=dropout)
        self.ESDmodel = ModelForTokenClassification(
            encoder=ESD_encoder,
            num_tags=ESD_num_tags,
            is_encoder_frozen=is_ESD_encoder_frozen,
            dropout=dropout)
        self.ratio = ratio
        self.crf = CRF(num_tags=num_tags, batch_first=True)

    def forward(self, inputs, W_e2n,labels=None, ESD_labels=None):
        logits, _ = self.NERmodel(inputs, labels)
        del inputs['pixel_values']
        ESD_logits, ESD_loss = self.ESDmodel(inputs, ESD_labels)
        batch_size=ESD_logits.shape[0]
        W_e2n=W_e2n.repeat(batch_size,1,1)
        logits += torch.bmm(ESD_logits, W_e2n)
        if labels==None or ESD_labels==None:
            return logits, None
        else:
            loss = self.crf(logits, labels, mask=inputs['attention_mask'].bool())
            loss = -1 * loss

            total_loss = loss + self.ratio * ESD_loss
            return logits, total_loss


class FlavaForNER(ModelForTokenClassification):
    def __init__(self, is_encoder_frozen:bool=True, dropout:float=0.1) -> None:
        encoder = FlavaModel.from_pretrained("facebook/flava-full")
        super().__init__(encoder=encoder,
                         num_tags=len(config.tag2id),
                         is_encoder_frozen=is_encoder_frozen,
                         dropout=dropout)


class FlavaForNERwithESD_bert_only(ModelForNERwithESD):
    def __init__(self,
                 ratio:float=1,
                 is_encoder_frozen:bool=True,
                 is_ESD_encoder_frozen:bool=True,
                 dropout:float=0.1) -> None:
        encoder = FlavaModel.from_pretrained("facebook/flava-full")
        ESD_encoder = FlavaTextModel.from_pretrained('facebook/flava-full')
        num_tags = len(config.tag2id)
        ESD_num_tags = len(config.ESD_id2tag)
        super().__init__(encoder, ESD_encoder, num_tags, ESD_num_tags, ratio,
                         is_encoder_frozen, is_ESD_encoder_frozen, dropout)

class ModelForNERwithESD_ComplexedVer(nn.Module):
    def __init__(self,
                 encoder,
                 ESD_encoder,
                 num_tags,
                 ESD_num_tags,
                 ratio,
                 is_encoder_frozen=True,
                 is_ESD_encoder_frozen=True,
                 dropout=0.1) -> None:
        super().__init__()
        self.encoder = encoder
        self.ESD_encoder = ESD_encoder
        self.crf = CRF(num_tags=num_tags, batch_first=True)
        self.ESD_crf = CRF(num_tags=ESD_num_tags, batch_first=True)
        self.classifier = nn.Linear(encoder.config.hidden_size, num_tags)
        self.ESD_classifier = nn.Linear(encoder.config.hidden_size,
                                        ESD_num_tags)
        self.ratio = ratio
        self.is_encoder_frozen = is_encoder_frozen
        self.is_ESD_encoder_frozen = is_ESD_encoder_frozen
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, labels, ESD_labels, W_e2n):
        if self.is_encoder_frozen:
            with torch.no_grad():
                hidden_states = self.encoder(**inputs).text_embeddings
        else:
            hidden_states = self.encoder(**inputs).text_embeddings

        hidden_states = self.dropout(hidden_states)

        logits = self.classifier(hidden_states)

        del inputs['pixel_values']

        if self.is_ESD_encoder_frozen:
            with torch.no_grad():
                ESD_hidden_states = self.ESD_encoder(
                    **inputs).last_hidden_state

        else:
            ESD_hidden_states = self.ESD_encoder(**inputs).last_hidden_state

        ESD_hidden_states = self.dropout(ESD_hidden_states)
        ESD_logits = self.ESD_classifier(ESD_hidden_states)
        ESD_loss = self.ESD_crf(ESD_logits, ESD_labels, mask=inputs['attention_mask'].bool())
        ESD_loss = -1 * ESD_loss
        batch_size=ESD_logits.shape[0]
        W_e2n=W_e2n.repeat(batch_size,1,1)
        logits += torch.bmm(ESD_logits, W_e2n)
        loss = self.crf(logits, labels, mask=inputs['attention_mask'].bool())
        loss = -1 * loss

        total_loss = loss + self.ratio * ESD_loss
        return logits, total_loss
