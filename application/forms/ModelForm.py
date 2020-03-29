"""
Contains choices for proposed detector and classifier architectures.
Automatically submits on select.
"""
from flask_wtf import FlaskForm
from wtforms import SelectField


class ModelForm(FlaskForm):
    detector = SelectField("Face Detector",
                           choices=[("mtcnn", "MTCNN"),
                                    ("casclas", "Cascade Classifier")],
                           default="mtcnn")
    
    classifier = SelectField("Face Classifier",
                             choices=[("mlp", "Multilayer Perceptron"),
                                      ("cnn", "CNN"),
                                      ("nb", "Categorical Naive Bayes"),
                                      ("svm", "Linear SVM")],
                             default="cnn")
