"""
Contains choices for proposed detector and classifier architectures.
Automatically submits on select.
""""
from flask_wtf import FlaskForm
from wtforms import SelectField


class ModelForm(FlaskForm):
    detector = SelectField("Face Detector",
                           choices=[("mtcnn", "MTCNN"),
                                    ("casclas", "Cascade Classifier")])
    
    classifier = SelectField("Face Classifier",
                             choices=[("mlp", "MLP"),
                                      ("cnn", "CNN")])
