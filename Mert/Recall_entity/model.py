import torch
from torch import nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from multi_encoder.model import MultiEncoderV2_2,MultiEncoderConfig
class LinkLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1=nn.Linear(768*2,768*2)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.3)
        self.fc2=nn.Linear(768*2,768)
        self.fc3=nn.Linear(768,128)
        self.classifier=nn.Linear(128,2)
    def forward(self,input):
        output=self.fc1(input)
        output=self.relu(output)
        output=self.dropout(output)
        output=self.fc2(output)
        output=self.relu(output)
        output=self.dropout(output)
        output=self.fc3(output)
        output=self.relu(output)
        output=self.dropout(output)
        output=self.classifier(output)
        return output

class ProjectLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1=nn.Linear(768,768*2)
        self.fc2=nn.Linear(768*2,768)
        self.relu=nn.ReLU()
    def forward(self,input):
        output=self.fc1(input)
        output=self.relu(output)
        output=self.fc2(output)
        output=self.relu(output)
        return output

class MertForEL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.multi_encoder=MultiEncoderV2_2.from_pretrained('/root/Multimodality-Link/Mert/MultiEncoder_ckpt/me2-flickr_1.pkl',MultiEncoderConfig())
        self.fc=nn.Linear(768,768)
        self.tanh=nn.Tanh()
        self.classifier=nn.Linear(768,2)
    def forward(self,**input):
        output=self.multi_encoder(**input).text_embeddings[:,0]
        output=self.fc(output)
        output=self.tanh(output)
        output=self.classifier(output)
        return output



