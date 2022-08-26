from argparse import ArgumentParser
import json
from flask import request

from server_utils import create_app, create_server
from wiki_service import query_entities

parser = ArgumentParser()
parser.add_argument("-a", "--address", help="The address of server", type=str, default="0.0.0.0")
parser.add_argument("-p", "--port", help="The port of server", type=int, default=3002)
args = parser.parse_args()
print(args)

app = create_app(__name__)


@app.route("/mert/wiki", methods=["POST"])
def query_controller():
    queries = json.loads(request.form.get("data"))
    results = query_entities(queries, app.logger)
    return json.dumps(results)


@app.route("/mert/test", methods=["GET"])
def query_test():
    return "TEST"


server = create_server(app, args.address, args.port)