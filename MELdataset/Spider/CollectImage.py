import wikipedia
import os
import json
import requests
import time


def img_download(img_url, img_name):
    # download an image from url
    name = img_name + '.' + img_url.split('.').pop()
    headers = {
        'User-Agent': 'Python/3.7 (718525108@qq.com) requests/2.23'
    }
    r = requests.get(img_url, headers=headers, stream=True)
    with open('drive/MyDrive/Images/' + name, 'wb') as file:
        file.write(r.content)
    return name


if __name__ == '__main__':
    all_files = os.listdir('drive/MyDrive/dataset_new')
    t0 = time.time()
    ok_file = open('drive/MyDrive/ok_entities.txt', 'r')
    ok_entities = ok_file.read()[:-1]  # 获得已经爬好的entity条目
    ok_file.close()
    pic_forms = {'jpg', 'jpeg', 'svg', 'png'}
    for file_name in all_files:
        with open('drive/MyDrive/dataset_new/' + file_name, 'r') as f:
            dataset = json.load(f)  # load dataset as dic
            f.close()
            for ind, data in enumerate(dataset.values()):         
                if ind < (int)(ok_entities):
                    continue
                wikidata_id = data['answer']
                try:
                    wikidata_json = requests.get(
                        'https://www.wikidata.org/w/api.php?action=wbgetentities&ids=' + wikidata_id + '&format=json&props=sitelinks').json()
                    title = wikidata_json['entities'][wikidata_id]['sitelinks']['enwiki']['title']
                    wikipedia_json = requests.get(
                        'https://en.wikipedia.org/w/api.php?action=query&titles=' + title + '&format=json').json()
                    pageid = ''
                    for value in wikipedia_json['query']['pages'].values():
                        pageid = value['pageid']
                    page = wikipedia.page(pageid=pageid)
                    brief = page.summary
                    imgs_url = page.images
                    data['brief'] = brief
                    img_list = []
                    for index, url in enumerate(imgs_url):
                        suffix = url.split('.').pop()
                        if url.find('commons') != -1 and (suffix in pic_forms):
                            img_list.append(img_download(url, wikidata_id + '_' + str(index)))
                    data['img_list'] = img_list
                    ok_file_wb = open('drive/MyDrive/ok_entities.txt', 'w')  # 这个负责写回已爬好的entity
                    ok_file_wb.write(str(ind) + '\n')
                    ok_file_wb.close()
                    with open('drive/MyDrive/dataset_new/' + file_name, 'w') as fb:  # 爬好一条保存一次
                        json.dump(dataset, fb)
                        fb.close()
                    if ind % 25 == 0:
                        t1 = time.time()
                        elapsed = str(t1 - t0)
                        print('Processing:' + str(ind) + ' of ' + str(
                            len(dataset.values())) + ' data in ' + file_name + ' elapsed:' + elapsed)
                except Exception as e:
                    print('catch error:', e)  # 若有异常，则记录异常的entity
                    with open('drive/MyDrive/ErrorEntities/' + file_name, 'a') as ef:
                        ef.write(wikidata_id + '\n')
                        ef.close()
