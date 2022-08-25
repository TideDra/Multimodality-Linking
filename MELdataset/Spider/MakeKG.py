from logging import Logger
from tqdm import tqdm
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from wikidata import WikiClient
import threading
from MakeMELv2 import is_human

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


def make_abstract(entity_id: str, client: WikiClient):
    entity = client.get(entity_id)
    if not is_human(entity_id):
        try:
            return entity['descriptions']['en']['value']
        except:
            return ""

    abstract = {}
    for pk, pv in property_ids.items():
        if not entity['claims'].get(pv):
            pk_value = ''
            for item in entity['claims'][pv]:
                if not item['mainsnak'].get('datavalue'):
                    pk_id = item['mainsnak']['datavalue']['value']['id']
                    pk_entity = client.get(pk_id)
                    if not pk_entity['labels'].get('en'):
                        if pk_value == '':
                            pk_value = pk_entity['labels']['en']['value']
                        else:
                            pk_value += ',' + pk_entity['labels']['en']['value']
            if pk_value != '':
                abstract[pk] = pk_value
    try:
        birthdata = entity['claims'][special_property_ids['BirthDate']
                                     ][0]['mainsnak']['datavalue']['value']['time'][1 :].split('-')[0]
    except:
        birthdata = ''
    try:
        birthplace_id = entity['claims'][special_property_ids['BirthPlace']][0]['mainsnak']['datavalue']['value']['id']
        birthplace_entity = client.get(birthplace_id)
        birthplace = birthplace_entity['labels']['en']['value']
    except:
        birthplace = ''
    if birthdata != '' or birthplace != '':
        abstract['Birth'] = birthdata + ',' + birthplace
    try:
        deathdata = entity['claims'][special_property_ids['DeathDate']
                                     ][0]['mainsnak']['datavalue']['value']['time'][1 :].split('-')[0]
    except:
        deathdata = ''
    try:
        deathplace_id = entity['claims'][special_property_ids['DeathPlace']][0]['mainsnak']['datavalue']['value']['id']
        deathplace_entity = client.get(deathplace_id)
        deathplace = deathplace_entity['labels']['en']['value']
    except:
        deathplace = ''
    if deathdata != '' or deathplace != '':
        abstract['Death'] = deathdata + ',' + deathplace
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
