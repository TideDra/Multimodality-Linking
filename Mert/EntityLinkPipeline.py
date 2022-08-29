from dataclasses import dataclass

import torch

from MNER.utils import NERpipeline


@dataclass
class ELPreprocessOutput:
    query_embedding: torch.Tensor
    entities: list


def EntityLinkPipeline_step1(query_text, query_image, multi_model, NER_model, multi_processor):
    '''First to extract embeddings and entities'''
    query_input = multi_processor(
        text=query_text, images=query_image, return_tensors="pt", padding="max_length", max_length=64, truncation=True
    )
    query_embedding = multi_model(**query_input).text_embeddings
    query_embedding = torch.squeeze(query_embedding, 0)
    NER_result = NERpipeline(model=NER_model, text=query_text, img=query_image)
    entities = NER_result[0]['entities']
    return ELPreprocessOutput(query_embedding, entities)


def EntityLinkPipeline_step2(preoutput: ELPreprocessOutput, entity_model, entity_processor, wikidata: list):
    '''Third to get answers with wikidata'''
    query_embedding = preoutput.query_embedding
    entities = preoutput.entities
    for idx, entity in enumerate(entities):
        mention_pos = entity['token_ids']
        entity_s, entity_e = mention_pos[0], mention_pos[-1]
        entity_embedding = torch.mean(query_embedding[entity_s : entity_e + 1], dim=0)

        api_results = wikidata[idx]
        candidate_ids = [cand['id'] for cand in api_results]
        candidate_abs = [cand['abs'] for cand in api_results]
        if len(candidate_abs) == 0:
            continue

        candidate_inputs = entity_processor(
            candidate_abs, return_tensors="pt", padding="max_length", max_length=64, truncation=True
        )
        candidate_embeddings = entity_model(**candidate_inputs).last_hidden_state[:, 0]
        entity_embedding_repeats = entity_embedding.repeat(len(candidate_embeddings), 1)
        similarities = torch.cosine_similarity(entity_embedding_repeats, candidate_embeddings)
        answer = candidate_ids[similarities.argmax()]
        entities[idx]['answer'] = answer
        #del entities[idx]['type']
        #del entities[idx]['token_ids']
    return entities