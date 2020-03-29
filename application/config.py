import os


class Config:
    SECRET_KEY = "hashtag-useful-file"
    PRETRAINED_CASCLAS = os.environ.get("PRETRAINED_CASCLAS") or\
                "application/static/haarcascade_frontalface_default.xml"
    MLP_WEIGHTS = os.environ.get("MLP_WEIGHTS") or\
                "application/static/weights/mlp"
    SVM = os.environ.get("SVM") or\
                "application/static/weights/linear_svm.pickle"
    CNN_WEIGHTS = os.environ.get("CNN_WEIGHTS") or\
                "application/static/weights/cnn"
    CATEGORICAL_NB = os.environ.get("CATEGORICAL_NB") or\
                "application/static/weights/categorical_nb.pickle"
    OFFLINE_IMG_URLS = os.environ.get("OFFLINE_IMG_URLS") or\
                "application/static/img_urls.txt"
