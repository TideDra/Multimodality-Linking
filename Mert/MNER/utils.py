from torch.utils.data import Dataset
from config import tag2id,ESD_tag2id
import numpy as np

class TwitterDataset(Dataset):
    def __init__(self,file_path) -> None:
        self.data=self.load_data(file_path)
    
    def load_data(self,file_path):
        dataset=open(file_path).readlines()
        Data={}
        idx=0
        ind=0
        end=len(dataset)
        self.W_e2n=np.zeros((len(ESD_tag2id),len(tag2id)))
        while ind < end:
            text=dataset[ind]
            if text[:5]=='IMGID':
                img_id=text[6:-1]
                ind+=1
                if dataset[ind][:2]=='RT':# skip name
                    ind+=3
                #read sentence
                sentence=''
                tags=[]
                ESD_tags=[]
                while ind < end:
                    text = dataset[ind]
                    if text=='\n':#reach the end of a sample
                        ind+=1
                        break
                    if text[:4]=='http':#skip url
                        ind+=1
                        continue
                    word,tag=text.replace('\n','').split('\t')
                    if sentence!='':#use space to split words
                        sentence+=' '
                        tags.append(0)
                        ESD_tags.append(0)
                    sentence+=word
                    #tag is mapped to char
                    tags+=[tag2id[tag]]*len(word)
                    ESD_tags+=[ESD_tag2id[tag[0]]]*len(word)
                    ind+=1
                Data[idx]={'imgid':img_id,'sentence':sentence,'tags':tags,'ESD_tags':ESD_tags}
                idx+=1
            else:
                ind+=1
        return Data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def get_W_e2n_():
    '''
    获取ESD到NER的转换矩阵
    '''