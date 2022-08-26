import logging
import sys
from pathlib import Path

from flask import Flask
from flask_cors import CORS
from gevent import pywsgi, signal

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
sys.path.append(str(root_path / "Mert"))


def create_app(name: str):
    app = Flask(name)
    CORS(app, resources="/*")
    app.logger.setLevel(logging.INFO)
    app.debug = True
    return app


def create_server(app: Flask, address: str, port: int):

    server = pywsgi.WSGIServer((address, port), app)
    server.serve_forever()

    def sig_handler(signum, frame):
        app.logger.info("Stop service")
        server.stop()

    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)

    return server
