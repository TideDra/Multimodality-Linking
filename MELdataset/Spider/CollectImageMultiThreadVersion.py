#!pip install wikipedia
import wikipedia
import os
import json
import requests
import time
from threading import Thread, Lock
import threading
from google.colab import drive

session = requests.Session()

def img_download(img_url, name):
    headers = {'User-Agent': 'Python/3.7 (2380433991@qq.com) requests/2.23'}
    r = session.get(img_url, headers=headers, stream=True)
    with open(pdir + '/Images/' + name, 'wb') as file:
        for a in r.iter_content(chunk_size=1024):
            file.write(a)
    #print("download", name)


def work():
    while True:
        lock.acquire()
        if mainfinished:
            print(f"{len(dllist)} images to be downloaded")
        if len(dllist) > 0:
            img_download(dllist[-1][0], dllist[-1][1])
            dllist.pop()
        elif mainfinished:
            break
        lock.release()
        time.sleep(1)
    print("Thread exited")


if __name__ == '__main__':
    drive.mount('/content/drive/', force_remount=True)
    #os.chdir("/content/drive/")
    #print(os.listdir('/content/drive/MyDrive/MELdataset'))
    pdir = '/content/drive/MyDrive/MELdataset'

    all_files = os.listdir(pdir + '/dataset')
    dllist = []
    t0 = time.time()

    if os.path.exists(pdir + '/dataset_new/imgdl.txt'):
        with open(pdir + '/dataset_new/imgdl.txt', 'r') as f:
            dllist = [s.strip().split('\t') for s in f.readlines()]
        #print(dllist)
        print(f"Read download list of length {len(dllist)}")

    mainfinished = False

    lock = Lock()
    for i in range(8):
        threading.Thread(target=work).start()

    imgext = ['.png', '.jpg', '.jpg', '.svg']

    for file_name in all_files:
        if file_name != "train.json":
            continue

        with open(pdir + '/dataset/' + file_name, 'r') as f:
            dataset = json.load(f)  # load dataset as dic

        dataset_new = {}
        if os.path.exists(pdir + '/dataset_new/' + file_name):
            with open(pdir + '/dataset_new/' + file_name, 'r') as f:
                dataset_new = json.load(f)

        nitems = len(dataset)
        for ind, data in enumerate(dataset.values()):
            if data["id"] in dataset_new:
                if (ind+1)%100==0:
                    print(f"skip {ind} {data['id']}")
                continue

            wikidata_id = data['answer']
            wikidata_json = session.get(
                'https://www.wikidata.org/w/api.php?action=wbgetentities&ids=' + wikidata_id +
                '&format=json&props=sitelinks'
            ).json()

            try:
                title = wikidata_json['entities'][wikidata_id]['sitelinks']['enwiki']['title']
                
                wikipedia_json = session.get(
                    'https://en.wikipedia.org/w/api.php?action=query&titles=' + title + '&format=json'
                ).json()
                pageid = ''
                for value in wikipedia_json['query']['pages'].values():
                    pageid = value['pageid']
                page = wikipedia.page(pageid=pageid)
                brief = page.summary
                imgs_url = page.images
                data['brief'] = brief
                img_list = []
                for index, url in enumerate(imgs_url):
                    if url.find('commons') != -1:
                        ext = url[url.rindex("."):].lower()
                        if not ext in imgext:
                            continue
                        fn = f'{wikidata_id}_{index}{ext}'
                        img_list.append(fn)
                        lock.acquire()
                        dllist.append([url, fn])
                        lock.release()

                data['img_list'] = img_list

                dataset_new[data["id"]] = data

            except:
                with open(pdir + '/dataset_new/err_entities.txt', 'a') as ferr:
                    print(json.dumps(data), file=ferr)
                    print(f"err: {ind} {json.dumps(data)}")
            
            if (ind + 1) % 10 == 0:
                t1 = time.time()
                lock.acquire()
                print(
                    f'Processing: {ind + 1}/{nitems} data in {file_name} elapsed: {t1 - t0}, with {len(dllist)} images to be downloaded'
                )
                with open(pdir + '/dataset_new/' + file_name, 'w') as fb:
                    json.dump(dataset_new, fb)
                with open(pdir + '/dataset_new/imgdl.txt', 'w') as f:
                    print('\n'.join([t[0] + '\t' + t[1] for t in dllist]), file=f)
                lock.release()

    mainfinished = True