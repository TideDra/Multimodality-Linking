
from tqdm.notebook import tqdm
import json
import sys
from urllib.parse import quote
from requests import Session

from time import sleep
import threading

class WikiClient:
    def __init__(self) -> None:
        self.wait_time = 3
        self.session = Session()

    def tryget(self, url: str):
        while True:
            try:
                response = self.session.get(url)
                response.encoding = "utf-8"
                return response.json()
            except:
                sleep(self.wait_time)

    def get(self, id: str = None, query: str = None):
        if id!=None:
            data = self.tryget(
                f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={id}&format=json&languages=en"
            )
            return data['entities'][id]
        if query != None:
            data = self.tryget(
                f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={quote(query)}&language=en&limit=10&format=json"
            )
            return data['search']


def make_desc(entity_id: str, client: WikiClient):
    entity = client.get(entity_id)
    try:
        return entity['descriptions']['en']['value']
    except:
        return ""

if __name__ == '__main__':
    with open(
        'KG.json', 'r'
    ) as f:
        dataset = json.load(f)
    datast_keys = []
    for k in dataset.keys():
        datast_keys.append(k)
    tbar = tqdm(total=len(datast_keys))

    class SpiderThread(threading.Thread):
        def __init__(self, threadID, name, s, e):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.name = name
            self.s = s
            self.e = e
            self.lock = threading.Lock()
            self.task = datast_keys[s : e]

        def run(self):
            client = WikiClient()

            for key in self.task:
                desc = make_desc(key, client)
                with self.lock:
                    dataset[key]['Description'] = desc
                    tbar.update(1)

    totalThread = 16  #需要创建的线程数，可以控制线程的数量

    lenList = len(datast_keys)  #列表的总长度
    gap = int(lenList / totalThread)  #列表分配到每个线程的执行数

    threadLock = threading.Lock()  #锁
    threads = []  #创建线程列表

    # 创建新线程和添加线程到列表
    for i in range(totalThread):
        thread = 'thread%s' % i
        if i == 0:
            thread = SpiderThread(0, "Thread-%s" % i, 0, gap)
        elif totalThread == i + 1:
            thread = SpiderThread(i, "Thread-%s" % i, i * gap, lenList)
        else:
            thread = SpiderThread(i, "Thread-%s" % i, i * gap, (i + 1) * gap)
        threads.append(thread)  # 添加线程到列表

    # 循环开启线程
    for i in range(totalThread):
        threads[i].start()

    # 等待所有线程完成
    for t in threads:
        t.join()
    with open('KGv2.json','w') as f:
      json.dump(dataset,f)