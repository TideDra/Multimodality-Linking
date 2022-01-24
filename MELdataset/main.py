import wptools
import wikipedia
import os
from urllib.request import urlretrieve
import json
from alive_progress import alive_it
from time import sleep


def img_download(img_url, img_name):
    # download an image from url
    name = 'Images/' + img_name + '.' + img_url[len(img_url) - 3:len(img_url)]
    urlretrieve(img_url, name)
    return name


if __name__ == '__main__':
    datasets_path = os.path.relpath('dataset')
    all_files = os.listdir(datasets_path)
    for file_name in all_files:
        with open('dataset/' + file_name, 'r') as f:
            dataset = json.load(f)  # load dataset as dic
            for data in alive_it(dataset.values()):
                wikidata_id = data['answer']
                wikidata = wptools.page(wikibase=wikidata_id)
                wikidata.get_wikidata(show=False)
                page_title = wikidata.data['title'].replace('_', ' ')
                page = wikipedia.page(title=page_title)
                brief = page.summary
                imgs_url = page.images
                data['brief'] = brief
                img_list = []
                for index, url in enumerate(imgs_url):
                    sleep(1)
                    if url.find('commons') != -1:
                        img_list.append(img_download(url, wikidata_id + '_' + str(index)))
                data['img_list'] = img_list
            with open('dataset_new/' + file_name, 'w') as fb:
                json.dump(dataset, fb)
