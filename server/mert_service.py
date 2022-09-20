from logging import Logger
import os
from typing import Dict
from PIL import Image
from concurrent import futures
import numpy as np

os.environ['TRANSFORMERS_CACHE'] = 'F:/Multimodality-Link/.cache/'

from transformers import logging as tlogging

tlogging.set_verbosity_error()

from transformers import FlavaProcessor
from Mert.EntityLinkPipeline import ELPreprocessOutput, EntityLinkPipeline_step1, EntityLinkPipeline_step2, EntityLinkPipelineV2
from Mert.MNER.model import MertForNERwithESD_bert_only
from Mert.multi_encoder.model import MultiEncoder
from Mert.common_config import PretrainedModelConfig
from Mert.Recall_entity.model import MertForEL


class MertService:
    def __init__(self, logger: Logger) -> None:
        self.logger = logger
        logger.info("Initialize models")
        mert_args = {"passes": ["mm"], "nhead": 4}
        self.NER_model = MertForNERwithESD_bert_only.from_pretrained(
            PretrainedModelConfig.nermodel_path, mert_config={
                "augment_text": False,
                **mert_args
            }
        ).eval()
        logger.info("NER_model ready")
        self.multi_model = MultiEncoder.from_pretrained(PretrainedModelConfig.multiencoder_path, **mert_args).eval()
        logger.info("multi_model ready")
        self.el_model = MertForEL.from_pretrained(PretrainedModelConfig.mertel_path, mert_config=mert_args).eval()
        logger.info("el_model ready")
        self.flava_processor = FlavaProcessor.from_pretrained('facebook/flava-full')
        logger.info("flava_processor ready")
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

    def MEL_step2V2(self, key: int, query_results: list, require_probs: bool=False):
        preoutput = self.pipeline_cache[key]
        entities = preoutput.entities
        del self.pipeline_cache[key]

        with futures.ThreadPoolExecutor() as executor:
            tasks = []
            for idx, q in enumerate(query_results):
                task = executor.submit(
                    EntityLinkPipelineV2,
                    query_text=preoutput.query_text,
                    query_img=preoutput.query_image,
                    candidate_abs=[cand["abs"] for cand in q],
                    model=self.el_model,
                    processor=self.flava_processor,
                    output_probs=require_probs,
                )
                tasks.append((task, idx))
            for task, idx in tasks:
                if require_probs:
                    ans, probs = task.result()
                    entities[idx]["answer"] = query_results[idx][ans]["id"]
                    entities[idx]["probs"] = probs
                    entities[idx]["rank"] = np.argsort(probs).tolist()
                else:
                    ans = task.result()
                    entities[idx]["answer"] = query_results[idx][ans]["id"]

        return entities
