from dataclasses import dataclass

import torch
from PIL import Image

from MNER.utils import NERpipeline


@dataclass
class ELPreprocessOutput:
    query_text: str
    query_image: Image
    query_embedding: torch.Tensor
    entities: list


@torch.no_grad()
def EntityLinkPipeline_step1(query_text, query_image, multi_model, NER_model, multi_processor):
    '''First to extract embeddings and entities'''
    query_input = multi_processor(
        text=query_text, images=query_image, return_tensors="pt", padding="max_length", max_length=64, truncation=True
    )
    query_embedding = multi_model(**query_input).text_embeddings
    query_embedding = torch.squeeze(query_embedding, 0)
    NER_result = NERpipeline(model=NER_model, text=query_text, img=query_image)
    entities = NER_result[0]['entities']
    return ELPreprocessOutput(query_text, query_image, query_embedding, entities)


@torch.no_grad()
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


@torch.no_grad()
def EntityLinkPipelineV2(query_text, query_img, candidate_abs, model, processor, output_probs: bool = False):
    '''
    Link query to one of the candidate

    Args:
      query_text(str):the query text.
      query_img(Image): the image of query_text
      candidate_abs(list of str): the abstracts of candidates, made by abs_dict_to_str().
      model: model for EL.
      processor: processor matched with the model. Default is FlavaProcessor.from_pretrained('facebook/flava-full') 
    '''
    text_input = [[query_text, candidate] for candidate in candidate_abs]
    img_input = [query_img] * len(text_input)
    multi_input = processor(
        text=text_input, images=img_input, return_tensors="pt", padding="max_length", max_length=160, truncation=True
    )
    logits: torch.Tensor = model(**multi_input)
    probs = logits[:, 0]
    if output_probs:
        return probs.argmax().item(), probs.tolist()
    else:
        return probs.argmax().item()