from time import sleep
import urllib.request as request
import json
from urllib.parse import quote
class WikiClient:
    def __init__(self) -> None:
        self.wait_time=3

    def get(self, id: str=None,query:str=None):
        assert id==None or query ==None,'Input either id or query, not both.'
        if id!=None:
            url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={id}&format=json&languages=en"
            while True:
                try:
                    response = request.urlopen(url)
                    break
                except:
                    sleep(self.wait_time)
            return json.loads(response.read().decode('utf-8'))['entities'][id]
        if query!=None:
            url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={quote(query)}&language=en&limit=10&format=json"
            while True:
                try:
                    response = request.urlopen(url)
                    break
                except:
                    sleep(self.wait_time)
            return json.loads(response.read().decode('utf-8'))['search']