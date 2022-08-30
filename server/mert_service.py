import os
from typing import Dict
from PIL import Image

os.environ['TRANSFORMERS_CACHE'] = 'F:/Multimodality-Link/.cache/'

from transformers import logging as tlogging
tlogging.set_verbosity_error()

from transformers import BertModel, BertTokenizer, FlavaProcessor
from Mert.EntityLinkPipeline import ELPreprocessOutput, EntityLinkPipeline_step1, EntityLinkPipeline_step2, EntityLinkPipelineV2
from Mert.MNER.model import MertForNERwithESD_bert_only
from Mert.multi_encoder.model import MultiEncoder
from Mert.common_config import PretrainedModelConfig
from Mert.Recall_entity.model import MertForEL

print("Initialize models")
NER_model = MertForNERwithESD_bert_only.from_pretrained(PretrainedModelConfig.nermodel_path).eval()
print("NER_model ready")
multi_model = MultiEncoder.from_pretrained(PretrainedModelConfig.multiencoder_path, forward_link=True).eval()
print("multi_model ready")
entity_model = BertModel.from_pretrained('bert-base-uncased').eval()
print("entity_model ready")
el_model = MertForEL.from_pretrained(PretrainedModelConfig.mertel_path).eval()
print("el_model ready")
flava_processor = FlavaProcessor.from_pretrained('facebook/flava-full')
print("flava_processor ready")
entity_processor = BertTokenizer.from_pretrained('bert-base-uncased')
print("entity_processor ready")
print("Service Init Complete")

g_querykey = 0
pipeline_cache: Dict[int, ELPreprocessOutput] = {}


def MEL_step1(caption: str, image: Image):
    '''First step: Obtain query entities from processor output.'''
    global g_querykey
    g_querykey += 1
    output = EntityLinkPipeline_step1(
        query_text=caption,
        query_image=image,
        multi_model=multi_model,
        NER_model=NER_model,
        multi_processor=flava_processor,
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


def MEL_step2V2(key: int, query_results: list):
    preoutput = pipeline_cache[key]
    entities = preoutput.entities
    del pipeline_cache[key]

    for idx, q in enumerate(query_results):
        ans = EntityLinkPipelineV2(
            query_text=preoutput.query_text,
            query_img=preoutput.query_image,
            candidate_abs=[cand["abs"] for cand in q],
            model=el_model,
            processor=flava_processor,
        )
        entities[idx]["answer"] = query_results[idx][ans]["id"]

    return entities
