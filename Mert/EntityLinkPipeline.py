import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
from MNER.utils import NERpipeline
from MELdataset.Spider.wikidata import WikiClient
from MELdataset.Spider.MakeKG import make_abstract
from Datasets.EntityLink_dataset import abs_dict_to_str


def EntityLinkPipeline(query_text, query_image, multi_model, NER_model, entity_model,
                       multi_processor, entity_processor):
    query_input = multi_processor(text=query_text,
                                  images=query_image,
                                  return_tensors="pt",
                                  padding="max_length",
                                  max_length=64,
                                  truncation=True)
    query_input=query_input.to('cpu')
    query_embedding = multi_model(**query_input).text_embeddings
    NER_result = NERpipeline(model=NER_model, text=query_text, img=query_image)
    entities = NER_result[0]['entities']
    client = WikiClient()

    for idx in range(len(entities)):    
        mention = entities[idx]['entity']
        mention_pos = entities[idx]['token_ids']
        entity_s, entity_e = mention_pos[0],mention_pos[-1]
        if entity_e>entity_s:
            entity_embedding = torch.mean(query_embedding[idx][entity_s:entity_e], dim=0)
        else:
            entity_embedding=query_embedding[idx][entity_s]
        api_results = client.get(query=mention)
        candidate_ids = []
        candidate_abs = []
        for cand in api_results:
            cand_id = cand['id']
            candidate_ids.append(cand_id)
            cand_abs = make_abstract(cand_id)
            cand_abs = abs_dict_to_str(cand_abs)
            candidate_abs.append(cand_abs)
        candidate_inputs = entity_processor(candidate_abs,
                                            return_tensors="pt",
                                            padding="max_length",
                                            max_length=64,
                                            truncation=True)
        
        candidate_embeddings=entity_model(**candidate_inputs).last_hidden_state[:,0]
        entity_embedding_repeats = entity_embedding.repeat(len(candidate_embeddings), 1)
        similarities = torch.cosine_similarity(entity_embedding_repeats,
                                                       candidate_embeddings)
        answer = candidate_ids[similarities.argmax()]
        entities[idx]['answer']=answer
        del entities[idx]['type']
        del entities[idx]['token_ids']
    return entities