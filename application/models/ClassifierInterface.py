"""
Unifies emotion recognition logic for all Classifier models.

Notes
-----
    1. All Classifier models have been trained/fine-tuned on FERPlus
        using the same label map, which is provided below.
        If you want to change one of the Classifiers weights,
        see the according class docstring.

"""
from abc import ABCMeta
from abc import abstractmethod
from typing import Dict
from typing import List

import numpy as np

LAB_MAP = {'neutral': 0,
           'happiness': 1,
           'surprise': 2,
           'sadness': 3,
           'anger': 4,
           'disgust': 5,
           'fear': 6,
           'contempt': 7,
           'unknown': 8}


class ClassifierInterface(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'preprocess_images') and 
                callable(subclass.preprocess_images) and 
                hasattr(subclass, 'predict_emotions') and 
                callable(subclass.predict_emotions) or 
                NotImplemented)


    @abstractmethod
    def preprocess_images(self, images: List[np.ndarray]) -> np.ndarray:
        """Preprocesses cropped images before prediction."""
        raise NotImplementedError


    @abstractmethod
    def predict_emotions(self, images: np.ndarray) -> List[Dict[str, float]]:
        """Predicts emotions in the images."""
        raise NotImplementedError
