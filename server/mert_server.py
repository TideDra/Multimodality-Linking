from argparse import ArgumentParser
import base64
from io import BytesIO

from flask import request, jsonify
from PIL import Image

from server_utils import create_app, create_server

from server.mert_service import MEL_step1, MEL_step2, MEL_step2V2
from server.wiki_service import query_entities

parser = ArgumentParser()
parser.add_argument("-a", "--address", help="The address of server", type=str, default="0.0.0.0")
parser.add_argument("-p", "--port", help="The port of server", type=int, default=5001)
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
    caption = request.json.get("caption", "")
    image=request.json.get("image", None)
    image = base64_pil(image) if image else Image.new("RGB", (32, 32), (0, 0, 0))
    app.logger.info(caption)
    result = MEL_step1(caption, image)
    app.logger.info(result)
    if args.wiki:
        query_result = query_entities(result["query"])
        app.logger.info(query_result)
        answer = MEL_step2V2(result["key"], query_result)
        app.logger.info(answer)
        result = {"answer": answer, "wikidata": query_result}

    return jsonify(result)


@app.route("/mert/back", methods=["POST"])
def query_controller_back():
    key = request.json.get("key", type=int)
    query_results = request.json.get("data")
    result = MEL_step2V2(key, query_results)
    return jsonify({"answer": result, "wikidata": query_results})


@app.route("/mert", methods=["GET"])
def query_test():
    return "MERT!"


server = create_server(app, args.address, args.port)