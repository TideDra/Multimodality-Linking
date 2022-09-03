from time import sleep
from urllib.parse import quote
from requests import Session


class WikiClient:
    def __init__(self) -> None:
        self.wait_time = 3
        self.query_limit = 5
        self.session = Session()

    def tryget(self, url: str):
        tries = 3
        while tries > 0:
            try:
                response = self.session.get(url)
                response.encoding = "utf-8"
                return response.json()
            except:
                tries -= 1
                sleep(self.wait_time)
        return None

    def get(self, id: str = None, query: str = None):
        assert id or query, 'Input either id or query, not both.'
        if id:
            data = self.tryget(
                f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={id}&format=json&languages=en"
            )
            return data['entities'][id]
        if query != None:
            data = self.tryget(
                f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={quote(query)}&language=en&limit={self.query_limit}&format=json"
            )
            return data['search']