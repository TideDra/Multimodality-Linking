from concurrent import futures
from typing import List
import Levenshtein

from MELdataset.Spider.MakeKG import make_abstract
from MELdataset.Spider.wikidata import WikiClient
from Mert.Datasets.EntityLink_dataset import abs_dict_to_str

client = WikiClient()


def filter_entity(query: str, cand: dict, entities: List[dict] = None):
    ...


def query_entities(queries: list, search_limit: int = None, mner_entities: List[dict] = None):
    def run(query:str, mner_entity: dict):
        def runrun(cand: dict):
            # 编辑距离筛除过长的
            if not (Levenshtein.ratio(cand["label"], query) > 0.2 or ("aliases" in cand and query in cand["aliases"])):
                return None
            entity = client.get(id=cand["id"])
            # 在类别是人的情况下，剔除不是人的
            if mner_entity and mner_entity["type"] == "PER" and 'P21' not in entity['claims']:
                return None
            return {
                **cand,
                "abs": abs_dict_to_str(make_abstract(entity, client)),
            }

        query_result = client.get(query=query, search_limit=search_limit)
        with futures.ThreadPoolExecutor(max_workers=256) as executor:
            if mner_entities:
                tasks = [
                    executor.submit(runrun, cand)
                    for cand in query_result
                ]
            else:
                tasks = [executor.submit(runrun, cand, None) for cand in query_result]
            results = [task.result() for task in tasks]
            results = [r for r in results if r]  # 筛掉None
        return results

    with futures.ThreadPoolExecutor(max_workers=256) as executor:
        tasks = [executor.submit(run, query, mner_entity) for query, mner_entity in zip(queries, mner_entities)]
        results = [task.result() for task in tasks]
    return results
