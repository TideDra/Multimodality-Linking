from concurrent import futures

from MELdataset.Spider.MakeKG import make_abstract
from MELdataset.Spider.wikidata import WikiClient
from Mert.Datasets.EntityLink_dataset import abs_dict_to_str

client = WikiClient()


def query_entities(queries: list):
    def run(query):
        def runrun(cand):
            return {
                **cand,
                "abs": abs_dict_to_str(make_abstract(cand['id'], client)),
            }

        query_result = client.get(query=query)
        with futures.ThreadPoolExecutor(max_workers=8) as executor:
            tasks = [executor.submit(runrun, cand) for cand in query_result]
            results = [task.result() for task in tasks]
        return results

    with futures.ThreadPoolExecutor(max_workers=8) as executor:
        tasks = [executor.submit(run, query) for query in queries]
        results = [task.result() for task in tasks]
    return results
