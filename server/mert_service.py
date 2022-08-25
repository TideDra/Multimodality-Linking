from Mert.EntityLinkPipeline import EntityLinkPipeline_step1, EntityLinkPipeline_step2

from PIL import Image
from Mert.multi_encoder.model import MultiEncoder
from Mert.MNER.model import MertForNERwithESD_bert_only
from transformers import logging, BertModel, BertTokenizer, FlavaProcessor

logging.set_verbosity_warning()

multi_model = MultiEncoder().eval()
NER_model = MertForNERwithESD_bert_only().eval()
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
