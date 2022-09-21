from logging import Logger
from tqdm import tqdm
import json
import sys
from pathlib import Path
from concurrent import futures

sys.path.append(str(Path(__file__).resolve().parent))
from wikidata import WikiClient
import threading

special_property_ids = {
    'BirthDate': 'P569',
    'BirthPlace': 'P19',
    'DeathDate': 'P570',
    'DeathPlace': 'P20',
}
property_ids = {
    'Sex': 'P21',
    'Religion': 'P140',
    'Occupation': 'P106',
    'Spouse': 'P26',
    'Languages': 'P1412',
    'Alma mater': 'P69'
}


def _get_birthdeath(entity, name: str, client: WikiClient):
    try:
        birthdata = entity['claims'][special_property_ids[f'{name}Date']
                                     ][0]['mainsnak']['datavalue']['value']['time'][1 :].split('-')[0]
    except:
        birthdata = ''

    try:
        birthplace_id = entity['claims'][special_property_ids[f'{name}Place']
                                         ][0]['mainsnak']['datavalue']['value']['id']
        birthplace_entity = client.get(id=birthplace_id)
        birthplace = birthplace_entity['labels']['en']['value']
    except:
        birthplace = ''

    return birthdata + ',' + birthplace if birthdata or birthplace else None


def make_abstract(entity: dict, client: WikiClient):
    if not 'P21' in entity['claims']:  # is not human
        try:
            return entity['descriptions']['en']['value']
        except:
            return ''

    abstract = {}
    with futures.ThreadPoolExecutor(max_workers=256) as executor:
        tasks = []
        for pk, pv in property_ids.items():
            if pv in entity['claims']:
                for item in entity['claims'][pv]:
                    if 'datavalue' in item['mainsnak']:
                        pk_id = item['mainsnak']['datavalue']['value']['id']
                        tasks.append((executor.submit(client.get, id=pk_id), pk))
        task_birth = executor.submit(_get_birthdeath, entity, 'Birth', client)
        task_death = executor.submit(_get_birthdeath, entity, 'Death', client)

        for task, pk in tasks:
            pk_entity = task.result()
            if 'en' in pk_entity['labels']:
                if not pk in abstract:
                    abstract[pk] = ''
                abstract[pk] += pk_entity['labels']['en']['value'] + ','
        if task_birth.result():
            abstract['Birth'] = task_birth.result()
        if task_death.result():
            abstract['Death'] = task_birth.result()

    return abstract


if __name__ == '__main__':
    with open(
        '/home/zero_lag/Document/srtp/Multimodality-Link/Mert/Recall_entity/data/Richpedia/Richpedia-MEL.json', 'r'
    ) as f:
        dataset = json.load(f)
    datast_keys = []
    for v in dataset.values():
        datast_keys.append(v['answer'])
        datast_keys += v['candidates'] if v.get('candidates') != None else []
    kg = {}
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
                abstract = make_abstract(key, client)
                with self.lock:
                    kg[key] = abstract
                    tbar.update(1)

    totalThread = 3  #需要创建的线程数，可以控制线程的数量

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
