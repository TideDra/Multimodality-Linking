from transformers import FlavaProcessor
import torch
class Config:
    id2tag={0:'O',1:'B-PER',2:'I-PER',3:'B-LOC',4:'I-LOC',5:'B-ORG',6:'I-ORG',7:'B-OTHER',8:'I-OTHER'}
    tag2id={'O':0,'B-PER':1,'I-PER':2,'B-LOC':3,'I-LOC':4,'B-ORG':5,'I-ORG':6,'B-OTHER':7,'I-OTHER':8}
    ESD_id2tag={0:'O',1:'B',2:'I'}
    ESD_tag2id={'O':0,'B':1,'I':2}

    processor=FlavaProcessor.from_pretrained("facebook/flava-full")

    special_token_tagid=0
    max_length=50
    test_text_path='./data/Twitter2015/test.txt'
    test_img_path='/home/zero_lag/Document/srtp/data/twitter2015/images/'

    train_text_path='/home/zero_lag/Document/srtp/Multimodality-Link/Mert/MNER/data/Twitter2015/train.txt'
    train_img_path='/home/zero_lag/Document/srtp/data/twitter2015/images/'

    dev_text_path='/home/zero_lag/Document/srtp/Multimodality-Link/Mert/MNER/data/Twitter2015/valid.txt'
    dev_img_path='/home/zero_lag/Document/srtp/data/twitter2015/images/'

    epochs=1
    learning_rate=1e-5
    batch_size=6
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path='/home/zero_lag/Document/srtp/Multimodality-Link/Mert/MNER/checkpoint/FlavaForNERwithESD_bert_only'
    max_checkpoint_num=2

    tb_dir='/home/zero_lag/Document/srtp/Multimodality-Link/Mert/MNER/tb_log'
config=Config()