import wikipedia
import os
import json
import requests
import time

def img_download(img_url, img_name):
    # download an image from url
    name = img_name + '.' + img_url[len(img_url) - 3:len(img_url)]
    headers = {
        'User-Agent': 'Python/3.7 (718525108@qq.com) requests/2.23'
    }
    r = requests.get(img_url, headers=headers, stream=False)
    with open('drive/MyDrive/Images/' + name, 'wb') as file:
        file.write(r.content)
    return name


if __name__ == '__main__':
    all_files = os.listdir('drive/MyDrive/dataset')
    t0 = time.time()
    for file_name in all_files:
        with open('drive/MyDrive/dataset/' + file_name, 'r') as f:
            dataset = json.load(f)  # load dataset as dic
            for ind,data in enumerate(dataset.values()):
                wikidata_id = data['answer']
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
                    if url.find('commons') != -1:
                        img_list.append(img_download(url, wikidata_id + '_' + str(index)))
                data['img_list'] = img_list
                if ind%50 == 0:
                  t1=time.time()
                  elapsed = str(t1-t0)
                  print('Processing:'+str(ind)+' of '+str(len(dataset.values()))+' data in '+file_name+' elapsed:'+elapsed)
            with open('drive/MyDrive/dataset_new/' + file_name, 'w') as fb:
                json.dump(dataset, fb)
