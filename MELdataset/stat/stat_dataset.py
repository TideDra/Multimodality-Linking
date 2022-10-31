from collections import Counter
import json
from pathlib import Path
import string


def count_words(s: str) -> int:
    # for c in string.punctuation:
    #     s = s.replace(c, " ")
    return len(s.split())


def stat_dataset(paths):
    if not isinstance(paths, list):
        paths = [paths]
    print("Stat", paths)
    data = {}
    for p in paths:
        data.update(json.loads(Path(p).read_text(encoding="utf8")))

    sentlens = []
    sentences = []
    candidates = []

    # 统计所有文本和候选
    for k, v in data.items():
        sentences.append(v["sentence"])
        if "candidates" in v:
            candidates.append(len(v["candidates"]))

    sentence_cnt = Counter(sentences)

    # 然后统计一下去重后的文本长度
    for k in sentence_cnt.keys():
        sentlens.append(count_words(k))

    sentlen_cnt = Counter(sentlens)
    candidate_cnt = Counter(candidates)

    print(f"Samples: {len(data)}")
    print(f"Mentions: {len(sentence_cnt)}  {len(sentences) / len(sentence_cnt)}")
    print(f"Candidates: {sum(candidates)}  {sum(candidates) / len(candidates)}")
    print(f"Text length: {sum(sentlens)/len(sentlens)}")
    print(json.dumps(sentlen_cnt))


stat_dataset([
    "MELdataset/KVQA/train_v2.json",
    "MELdataset/KVQA/dev_v2.json",
    "MELdataset/KVQA/test_v2.json",
])

stat_dataset([
    "MELdataset/newRichpedia/train.json",
    "MELdataset/newRichpedia/dev.json",
    "MELdataset/newRichpedia/test.json",
])
# stat_dataset("MELdataset/RichpediaMEL/Richpedia-MELv2.json")
# stat_dataset("MELdataset/RichpediaMEL/Richpedia-MELv2_train.json")
# stat_dataset("MELdataset/RichpediaMEL/Richpedia-MELv2_val.json")
# stat_dataset("MELdataset/RichpediaMEL/Richpedia-MELv2_test.json")
# stat_dataset("MELdataset/RichpediaMEL/Richpedia-MEL-raw.json")