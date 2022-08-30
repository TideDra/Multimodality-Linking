from logging import Logger
import os
from typing import Dict
from PIL import Image

os.environ['TRANSFORMERS_CACHE'] = 'F:/Multimodality-Link/.cache/'

from transformers import logging as tlogging

tlogging.set_verbosity_error()

from transformers import BertModel, BertTokenizerFast, FlavaProcessor
from Mert.EntityLinkPipeline import ELPreprocessOutput, EntityLinkPipeline_step1, EntityLinkPipeline_step2, EntityLinkPipelineV2
from Mert.MNER.model import MertForNERwithESD_bert_only
from Mert.multi_encoder.model import MultiEncoder
from Mert.common_config import PretrainedModelConfig
from Mert.Recall_entity.model import MertForEL


class MertService:
    def __init__(self, logger: Logger) -> None:
        self.logger = logger
        logger.info("Initialize models")
        self.NER_model = MertForNERwithESD_bert_only.from_pretrained(PretrainedModelConfig.nermodel_path).eval()
        logger.info("NER_model ready")
        self.multi_model = MultiEncoder.from_pretrained(PretrainedModelConfig.multiencoder_path).eval()
        logger.info("multi_model ready")
        self.entity_model = BertModel.from_pretrained('bert-base-uncased').eval()
        logger.info("entity_model ready")
        self.el_model = MertForEL.from_pretrained(PretrainedModelConfig.mertel_path).eval()
        logger.info("el_model ready")
        self.flava_processor = FlavaProcessor.from_pretrained('facebook/flava-full')
        logger.info("flava_processor ready")
        self.entity_processor = BertTokenizerFast.from_pretrained('bert-base-uncased')
        logger.info("entity_processor ready")
        logger.info("Service Init Complete")

        self.g_querykey = 0
        self.pipeline_cache: Dict[int, ELPreprocessOutput] = {}

    def MEL_step1(self, caption: str, image: Image):
        '''First step: Obtain query entities from processor output.'''
        self.g_querykey += 1
        output = EntityLinkPipeline_step1(
            query_text=caption,
            query_image=image,
            multi_model=self.multi_model,
            NER_model=self.NER_model,
            multi_processor=self.flava_processor,
        )
        self.pipeline_cache[self.g_querykey] = output
        query = [entity['entity'] for entity in output.entities]
        return {"key": self.g_querykey, "query": query}

    def MEL_step2(self, key: int, queries: list):
        result = EntityLinkPipeline_step2(
            preoutput=self.pipeline_cache[key],
            entity_model=self.entity_model,
            entity_processor=self.entity_processor,
            wikidata=queries,
        )
        del self.pipeline_cache[key]
        return result

    def MEL_step2V2(self, key: int, query_results: list):
        preoutput = self.pipeline_cache[key]
        entities = preoutput.entities
        del self.pipeline_cache[key]

        for idx, q in enumerate(query_results):
            ans = EntityLinkPipelineV2(
                query_text=preoutput.query_text,
                query_img=preoutput.query_image,
                candidate_abs=[cand["abs"] for cand in q],
                model=self.el_model,
                processor=self.flava_processor,
            )
            entities[idx]["answer"] = query_results[idx][ans]["id"]

        return entities
