import json
from pathlib import Path
from typing import Any, List


def rearrange(lst: List[Any], order: List[int]):
    return [lst[i] for i in order]


filename = "test_result - morex3.json"
input_file = Path(r"E:\学校的的的的\对比学习SRTP\Mert-MEL-test\autotest") / filename
with open(input_file, "r", encoding="utf-8") as fin:
    td = json.load(fin)

disp = []
for tc in td:
    disp.append([])
    for i, a in enumerate(tc["result"]["answer"]):
        if "rank" not in a:
            continue
        rank = a["rank"][::-1]
        a["rank"] = list(range(len(rank)))
        a["probs"] = rearrange(a["probs"], rank)
        tc["result"]["wikidata"][i] = rearrange(tc["result"]["wikidata"][i], rank)
        disp[-1].append(
            ''.join([
                f"{j+1}. {w['label']}\t{p*100:.1f}\n  ({w['description'] if 'description' in w else ''})\n"
                for j, p, w in zip(range(len(rank)), a["probs"], tc["result"]["wikidata"][i])
            ])
        )
    disp[-1] = '\n'.join(disp[-1])
disp = '\n'.join(disp)

with open(input_file.with_name(filename.replace(".json", " - rea.json")), "w", encoding="utf-8") as fout:
    json.dump(td, fout, ensure_ascii=False, indent=2)

with open(input_file.with_name(filename.replace(".json", " - rea.txt")), "w", encoding="utf-8") as fout:
    fout.write(disp)