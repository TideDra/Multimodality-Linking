from transformers import FlavaProcessor
import torch
class Config:
    id2tag={0:'O',1:'B-PER',2:'I-PER',3:'B-LOC',4:'I-LOC',5:'B-ORG',6:'I-ORG',7:'B-MISC',8:'I-MISC'}
    tag2id={'O':0,'B-PER':1,'I-PER':2,'B-LOC':3,'I-LOC':4,'B-ORG':5,'I-ORG':6,'B-MISC':7,'I-MISC':8}
    b2m={1:2,3:4,5:6,7:8}
    ESD_id2tag={0:'O',1:'B',2:'I'}
    ESD_tag2id={'O':0,'B':1,'I':2}

    processor=FlavaProcessor.from_pretrained("facebook/flava-full")

    special_token_tagid=0
    max_length=128
    test_text_path='/home/zero_lag/Document/srtp/Multimodality-Link/Mert/MNER/data/Twitter2017/test.txt'
    test_img_path='/home/zero_lag/Document/srtp/Multimodality-Link/Mert/MNER/data/Twitter2017/images'
    test_cache_path='/home/zero_lag/Document/srtp/Multimodality-Link/Mert/MNER/data/Twitter2017/test_cache'

    train_text_path='/home/zero_lag/Document/srtp/Multimodality-Link/Mert/MNER/data/Twitter2017/train.txt'
    train_img_path='/home/zero_lag/Document/srtp/Multimodality-Link/Mert/MNER/data/Twitter2017/images'
    train_cache_path='/home/zero_lag/Document/srtp/Multimodality-Link/Mert/MNER/data/Twitter2017/train_cache'

    dev_text_path='/home/zero_lag/Document/srtp/Multimodality-Link/Mert/MNER/data/Twitter2017/valid.txt'
    dev_img_path='/home/zero_lag/Document/srtp/Multimodality-Link/Mert/MNER/data/Twitter2017/images'
    dev_cache_path='/home/zero_lag/Document/srtp/Multimodality-Link/Mert/MNER/data/Twitter2017/dev_cache'

    shuffle_seed=10086
    num_workers=1
    epochs=10
    learning_rate=5*1e-5
    batch_size=2
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MultiEncoder_no_MCA_model_path='/home/zero_lag/Document/srtp/exp_data/MultiEncoder_pretrained/me-flickr_2.pkl'
    checkpoint_path='/home/zero_lag/Document/srtp/Multimodality-Link/Mert/MNER/checkpoint'
    max_checkpoint_num=2

    tb_dir='/home/zero_lag/Document/srtp/Multimodality-Link/Mert/MNER/tb_log'

config=Config()

class BertBiLSTMEncoderConfig:
    blstm_hidden_size=768
    blstm_dropout=0.4
    hidden_size=384
    is_bert_frozen=False

BertBiLSTMEncoderConfigforFNEBB=BertBiLSTMEncoderConfig()