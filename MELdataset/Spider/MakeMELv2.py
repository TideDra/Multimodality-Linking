from wikidata import WikiClient
import json
import random
from tqdm import tqdm
import threading

def is_human(id):
    client=WikiClient()
    entity=client.get(id=id)
    if entity['claims'].get('P21')!=None:
        return True
    else:
        return False
if __name__=='__main__':
    with open('/home/zero_lag/Document/srtp/Multimodality-Link/MELdataset/RichpediaMEL/Richpedia-MEL.json') as f:
        dataset=json.load(f)
    
    
    datast_keys=list(dataset.keys())
    tbar=tqdm(total=len(datast_keys))
    
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
            client=WikiClient()
            for key in self.task:
                mention=dataset[key]['mentions']
                api_results=client.get(query=mention)
                candidates=[]
                for cand in api_results:
                    cand_id=cand['id']
                    if cand_id!=dataset[key]['answer'] and is_human(cand_id):
                        candidates.append(cand_id)
                if len(candidates)==0:
                    random_cands=random.sample(dataset.keys(),2)
                    random_cand=dataset[random_cands[0]]['answer'] if random_cands[0]!=key else dataset[random_cands[1]]['answer']
                    candidates.append(random_cand)
                with self.lock:
                    dataset[key]['candidates']=candidates
                    tbar.update(1)
    
    totalThread = 16 #需要创建的线程数，可以控制线程的数量
    
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