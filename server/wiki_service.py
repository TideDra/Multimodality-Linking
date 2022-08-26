from MELdataset.Spider.MakeKG import make_abstract
from MELdataset.Spider.wikidata import WikiClient
from Mert.Datasets.EntityLink_dataset import abs_dict_to_str

client = WikiClient()


def query_entities(queries: list, logger=None):
    results = []
    for query in queries:
        if logger: logger.info(query)
        query_result = client.get(query=query)
        results.append([{
            **cand,
            "abs": abs_dict_to_str(make_abstract(cand['id'], client)),
        } for cand in query_result])
    return results