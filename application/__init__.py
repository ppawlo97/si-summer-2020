import logging
logging.basicConfig(level=logging.INFO)

from flask import Flask

from application.config import Config


app = Flask(__name__)
app.config.from_object(Config)

from application.models.classifiers.CNNClassifier import CNNClassifier
from application.models.classifiers.MLPClassifier import MLPClassifier
from application.models.classifiers.NaiveBayesClassifier import NaiveBayesClassifier
from application.models.classifiers.SVMClassifier import SVMClassifier
from application.models.detectors.CasClasDetector import CasClasDetector
from application.models.detectors.MTCNNDetector import MTCNNDetector
from application.utils import get_urls_list


logging.info("Loading models...")
MODELS = {"mtcnn": MTCNNDetector(),
          "casclas": CasClasDetector(app.config["PRETRAINED_CASCLAS"]),
          "mlp": MLPClassifier(app.config["MLP_WEIGHTS"])}

IMG_URLS = get_urls_list(app.config["OFFLINE_IMG_URLS"])

from application import routes
