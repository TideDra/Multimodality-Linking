from tqdm import tqdm
import json
import urllib.request as request
import threading
from time import time
with open(
        '/home/zero_lag/Document/srtp/Multimodality-Link/Mert/Recall_entity/data/Richpedia/Richpedia-MEL.json',
        'r') as f:
    dataset = json.load(f)
datast_keys=list(dataset.keys())
kg = {}
tbar=tqdm(total=len(datast_keys))

class WikiClient:
    def __init__(self) -> None:
        self.wait_time=0.5
        self.last_request_time=time()

    def get(self, id: str):
        while(time()-self.last_request_time<self.wait_time):
            pass
        url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={id}&format=json&languages=en"
        self.last_request_time=time()
        response = request.urlopen(url)
        return json.loads(response.read().decode('utf-8'))['entities'][id]

class SpiderThread(threading.Thread):
    def __init__(self, threadID, name, s, e):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.s = s
        self.e = e
        self.lock = threading.Lock()
        self.task=datast_keys[s:e]
    def run(self):
        client = WikiClient()
        special_property_ids = {
            'BirthDate': 'P569',
            'BirthPlace': 'P19',
            'DeathDate': 'P570',
            'DeathPlace': 'P20'
        }
        property_ids = {
            'Sex': 'P21',
            'Religion': 'P140',
            'Occupation': 'P106',
            'Spouse': 'P26',
            'Languages': 'P1412',
            'Alma mater': 'P69'
        }
        for key in self.task:
            entity = client.get(key)
            abstract = {}
            for pk, pv in property_ids.items():
                if entity['claims'].get(pv) != None:
                    pk_value = ''
                    for item in entity['claims'][pv]:
                        if item['mainsnak'].get('datavalue') != None:
                            pk_id = item['mainsnak']['datavalue']['value'][
                                'id']
                            pk_entity = client.get(pk_id)
                            if pk_entity['labels'].get('en') != None:
                                if pk_value == '':
                                    pk_value = pk_entity['labels']['en'][
                                        'value']
                                else:
                                    pk_value += ',' + pk_entity['labels'][
                                        'en']['value']
                    if pk_value != '':
                        abstract[pk] = pk_value
            try:
                birthdata = entity['claims'][
                    special_property_ids['BirthDate']][0]['mainsnak'][
                        'datavalue']['value']['time'][1:].split('-')[0]
            except:
                birthdata = ''
            try:
                birthplace_id = entity['claims'][special_property_ids[
                    'BirthPlace']][0]['mainsnak']['datavalue']['value']['id']
                birthplace_entity = client.get(birthplace_id)
                birthplace = birthplace_entity['labels']['en']['value']
            except:
                birthplace = ''
            if birthdata != '' or birthplace != '':
                abstract['Birth'] = birthdata + ',' + birthplace
            try:
                deathdata = entity['claims'][
                    special_property_ids['DeathDate']][0]['mainsnak'][
                        'datavalue']['value']['time'][1:].split('-')[0]
            except:
                deathdata = ''
            try:
                deathplace_id = entity['claims'][special_property_ids[
                    'DeathPlace']][0]['mainsnak']['datavalue']['value']['id']
                deathplace_entity = client.get(deathplace_id)
                deathplace = deathplace_entity['labels']['en']['value']
            except:
                deathplace = ''
            if deathdata != '' or deathplace != '':
                abstract['Death'] = deathdata + ',' + deathplace
            with self.lock:
                kg[key] = abstract
                tbar.update(1)

totalThread = 3 #需要创建的线程数，可以控制线程的数量

lenList = len(datast_keys) #列表的总长度
gap = int(lenList / totalThread) #列表分配到每个线程的执行数

threadLock = threading.Lock() #锁
threads = [] #创建线程列表

# 创建新线程和添加线程到列表
for i in range(totalThread):
   thread = 'thread%s' % i
   if i == 0:
       thread = SpiderThread(0, "Thread-%s" % i, 0,gap)
   elif totalThread==i+1:
       thread = SpiderThread(i, "Thread-%s" % i, i*gap,lenList)
   else:
       thread = SpiderThread(i, "Thread-%s" % i, i*gap,(i+1)*gap)
   threads.append(thread) # 添加线程到列表

# 循环开启线程
for i in range(totalThread):
   threads[i].start()

# 等待所有线程完成
for t in threads:
   t.join()