import logging
logging.basicConfig(level=logging.INFO)

from flask import Flask

from application.config import Config


app = Flask(__name__)
app.config.from_object(Config)

from application.models.CasClasDetector import CasClasDetector
from application.models.MLPClassifier import MLPClassifier
from application.models.MTCNNDetector import MTCNNDetector

logging.info("Loading models...")
MODELS = {"mtcnn": MTCNNDetector(),
          "casclas": CasClasDetector(app.config["PRETRAINED_CASCLAS"]),
          "mlp": MLPClassifier(app.config["MLP_WEIGHTS"])}


from application import routes
