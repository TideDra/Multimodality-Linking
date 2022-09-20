from time import sleep
from requests import Session


class WikiClient:
    def __init__(self, query_limit: int = 10) -> None:
        self.wait_time = 3
        self.query_limit = query_limit
        self.session = Session()
        self.api = "https://www.wikidata.org/w/api.php"

    def __tryget(self, params: dict):
        params = {"format": "json", **params}
        tries = 3
        while tries > 0:
            try:
                response = self.session.get(self.api, params=params)
                response.encoding = "utf-8"
                return response.json()
            except:
                tries -= 1
                sleep(self.wait_time)
        return None

    def get(self, *, id: str = None, query: str = None, search_limit: int = None):
        assert id or query, 'Input either id or query, not both.'
        if id:
            data = self.__tryget(params={"action": "wbgetentities", "ids": id, "languages": "en"})
            return data['entities'][id]
        if query:
            data = self.__tryget(
                params={
                    "action": "wbsearchentities",
                    "search": query,
                    "language": "en",
                    "limit": search_limit or self.query_limit
                }
            )
            return data['search']