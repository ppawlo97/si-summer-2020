import os


class Config:
    SECRET_KEY = "hashtag-useful-file"
    PRETRAINED_CASCLAS = os.environ.get("PRETRAINED_CASCLAS") or\
                "application/static/haarcascade_frontalface_default.xml"
