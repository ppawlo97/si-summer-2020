import logging
logging.basicConfig(level=logging.INFO)

from flask import Flask

from application.config import Config


app = Flask(__name__)
app.config.from_object(Config)

from application.models.MTCNNDetector import MTCNNDetector
from application.models.CasClasDetector import CasClasDetector

logging.info("Loading models...")
MODELS = {"mtcnn": MTCNNDetector(),
          "casclas": CasClasDetector(app.config["PRETRAINED_CASCLAS"])}


from application import routes
