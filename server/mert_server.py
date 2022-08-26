from argparse import ArgumentParser
import base64
import json
from io import BytesIO

from flask import request
from PIL import Image

from server_utils import create_app, create_server

from server.mert_service import MEL_step1, MEL_step2
from server.wiki_service import query_entities

parser = ArgumentParser()
parser.add_argument("-a", "--address", help="The address of server", type=str, default="0.0.0.0")
parser.add_argument("-p", "--port", help="The port of server", type=int, default=3001)
parser.add_argument("-w", "--wiki", help="Access Wikidata in this server", action="store_true")
args = parser.parse_args()


def base64_pil(base64_str: str):
    image = base64.b64decode(base64_str.split("base64,")[1])
    image = BytesIO(image)
    image = Image.open(image)
    return image


app = create_app(__name__)


@app.route("/mert/query", methods=["POST"])
def query_controller():
    caption = request.form.get("caption", "")
    image = base64_pil(request.form.get("image")) if "image" in request.form else Image.new("RGB", (32, 32), (0, 0, 0))
    result = MEL_step1(caption, image)
    if args.wiki:
        query_result = query_entities(result["query"])
        result = MEL_step2(result["key"], query_result)
    return json.dumps(result)


@app.route("/mert/back", methods=["POST"])
def query_controller_back():
    key = request.form.get("key", type=int)
    query_results = json.loads(request.form.get("data"))
    result = MEL_step2(key, query_results)
    return json.dumps(result)


@app.route("/mert", methods=["GET"])
def query_test():
    return "MERT!"


server = create_server(app, args.address, args.port)