import os

os.environ['TRANSFORMERS_CACHE'] = 'F:/Multimodality-Link/.cache/'

from Mert.EntityLinkPipeline import EntityLinkPipeline_step1, EntityLinkPipeline_step2

from PIL import Image
from Mert.multi_encoder.model import MultiEncoderV2_2
from Mert.MNER.model import MertForNERwithESD_bert_only
from transformers import logging, BertModel, BertTokenizer, FlavaProcessor
import torch

logging.set_verbosity_warning()

NER_model = MertForNERwithESD_bert_only.from_pretrained(
    r"F:\Multimodality-Link\.ckpt\MertForNERwithESD_bert_only_epoch_44_macrof1_81.089_microf1_85.714_1661235521.bin"
).eval()
multi_model = MultiEncoderV2_2.from_pretrained("F:/Multimodality-Link/.ckpt/me2-flickr_1.pkl").eval()
entity_model = BertModel.from_pretrained('bert-base-uncased').eval()
multi_processor = FlavaProcessor.from_pretrained('facebook/flava-full')
entity_processor = BertTokenizer.from_pretrained('bert-base-uncased')

g_querykey = 0
pipeline_cache = {}


def MEL_step1(caption: str, image: Image):
    '''First step: Obtain query entities from processor output.'''
    global g_querykey
    g_querykey += 1
    output = EntityLinkPipeline_step1(
        query_text=caption,
        query_image=image,
        multi_model=multi_model,
        NER_model=NER_model,
        multi_processor=multi_processor,
    )
    pipeline_cache[g_querykey] = output
    query = [entity['entity'] for entity in output.entities]
    return {"key": g_querykey, "query": query}


def MEL_step2(key: int, queries: list):
    result = EntityLinkPipeline_step2(
        preoutput=pipeline_cache[key],
        entity_model=entity_model,
        entity_processor=entity_processor,
        wikidata=queries,
    )
    del pipeline_cache[key]
    return result
