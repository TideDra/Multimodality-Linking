import json
from pathlib import Path
from requests import Session
from PIL import Image
from io import BytesIO
import base64


def image_to_base64(image: Image.Image, fmt='png') -> str:
    output_buffer = BytesIO()
    image.save(output_buffer, format=fmt)
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return f'data:image/{fmt};base64,' + base64_str


base_dir = Path(r"E:\学校的的的的\对比学习SRTP\Mert-MEL-test\autotest")
with open(base_dir / "totest.json", "r", encoding="utf8") as fin:
    td = json.load(fin)

ses = Session()

for idx, item in enumerate(td):
    print(idx)
    img = Image.open(base_dir / item["image"])
    res = ses.post(
        "http://127.0.0.1:5001/mert/query",
        json={
            "image": image_to_base64(img),
            "caption": item["text"],
            "require_probs": True,
            "search_limit": 20,
        }
    )
    print(res.json())
    item["result"] = res.json()

    with open(base_dir / "test_result.json", "w", encoding="utf8") as fout:
        json.dump(td, fout, ensure_ascii=False)
