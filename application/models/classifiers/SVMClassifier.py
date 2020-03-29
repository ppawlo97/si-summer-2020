"""
Support Vector Machines trained with SGD.

Notes
-----
    1. SVM used in the project can be found in application/static/weights/linear_svm.pickle.
    2. If you want change the default LAB_MAP then application's __init__.py is
        the proper place to do it.

"""
import os
import pickle
from typing import Dict
from typing import List

import cv2
import imutils
import numpy as np
import tensorflow as tf

from application.models.classifiers.ClassifierInterface import ClassifierInterface 
from application.models.classifiers.ClassifierInterface import LAB_MAP


class SVMClassifier(ClassifierInterface):
    def __init__(self,
                 svm_path: str,
                 lab_map: Dict[str, int] = None):
        
        if not os.path.exists(svm_path):
            raise FileNotFoundError("Pickle file with\n\
                SVM does not exist!")

        if lab_map is None:
            self.lab2idx = LAB_MAP
        else:
            self.lab2idx = lab_map
        
        self.idx2lab = {i: lab for lab, i in self.lab2idx.items()}
        with open(svm_path, "rb") as mdl:
            self.svm = pickle.load(mdl)
        self.name = "svm"


    def preprocess_images(self, images: List[np.ndarray]) -> tf.Tensor:
        """Overrides ClassifierInterface.preprocess_images()"""
        grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                    for img in images]
        resized = [imutils.resize(img, width=48, height=48)
                        for img in grays]
        res = tf.stack([tf.image.resize_with_crop_or_pad(tf.expand_dims(img,
                                                                        axis=-1),
                                                        48, 48)
                            for img in resized], axis=0)
        res = tf.dtypes.cast(tf.reshape(res, (-1, 48*48)),
                             dtype=tf.float32)
        return res


    def predict_emotions(self, images: tf.Tensor) -> List[Dict[str, float]]:
        """Overrides ClassifierInterface.predict_emotions()"""
        preds = self.svm.predict(images)
        res = [{self.idx2lab[pred]: 1} for pred in preds]
        return res
