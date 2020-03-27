"""
Multilayer Perceptron.

Notes
-----
    1. MLP weights used in the project can be found in application/static/weights/mlp.
    2. Exact architecture of the MLP used in the project with the implementation
        can found at the bottom of the file. Use default parameters to obtain the same
        model as used in the project.
    3. If you want change the default LAB_MAP then application's __init__.py is
        the proper place to do it. 

"""
import os
from typing import Dict
from typing import List

import cv2
import imutils
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten

from application.models.ClassifierInterface import ClassifierInterface
from application.models.ClassifierInterface import LAB_MAP
from application.utils import center_crop


class MLPClassifier(ClassifierInterface):
    def __init__(self,
                 mlp_path: str,
                 lab_map: Dict[str, int] = None):
        
        if not os.path.exists(mlp_path):
            raise FileNotFoundError("File with MLP\n\
                weights does not exists!")

        if lab_map is None:
            self.lab2idx = LAB_MAP
        else:
            self.lab2idx = lab_map

        self.idx2lab = {i: lab for lab, i in self.lab2idx.items()}
        self.model = tf.keras.models.load_model(mlp_path, compile=False)
        self.name = "mlp"


    def preprocess_images(self, images: List[np.ndarray]) -> np.ndarray:
        """Overrides ClassifierInterface.preprocess_images()"""
        grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                    for img in images]
        resized = [imutils.resize(img, width=48, height=48)
                        for img in grays]
        res = tf.stack([center_crop(img, 48, 48) 
                            for img in resized], axis=0)
        res = tf.dtypes.cast(res/255, dtype=tf.float32)
        
        if res.ndim == 2:
            res = tf.expand_dims(res, axis=[0, -1])
        else:
            res = tf.expand_dims(res, axis=[-1])
            
        return res


    def predict_emotions(self, images: np.ndarray) -> List[Dict[str, float]]:
        """Overrides ClassifierInterface.predict_emotions()"""
        preds = self.model(images)
        res = [{self.idx2lab[ix]: p.numpy() for ix, p in enumerate(pred)}
                                                for pred in preds]
        return res


class MLP(tf.keras.Model):
    def __init__(self,
                 num_labels: int,
                 num_dense_units: int = 256,
                 dropout_rate: float = 0.3,
                 num_blocks: int = 3,
                 name: str= "mlp",
                 **kwargs):
        super(MLP, self).__init__(name=name, **kwargs)
        
        if num_blocks < 1:
            raise Exception("Cannot have less than 1 block!")
        self.num_blocks = num_blocks
        
        self.flatten = Flatten(name="flatten")
        
        for num in range(1, num_blocks + 1):
            setattr(self, "dense_{}".format(num), Dense(num_dense_units,
                                                        activation="relu",
                                                        name="dense_{}".format(num)))
            setattr(self, "batch_norm_{}".format(num), BatchNormalization(name="batch_norm_{}".format(num)))
        
        self.dropout = Dropout(dropout_rate,
                               name="dropout")
        self.classifier = Dense(num_labels,
                                activation="softmax",
                                name="classifier")
        
        
    def build(self, inputs_shape):
        self.flatten.build(inputs_shape)
        self.dense_1.build((None, tf.math.reduce_prod(inputs_shape[1:])))
        for layer in self.layers:
            if layer.name not in ["flatten", "dense_1"]:
                layer.build((None, self.dense_1.units))
        super(MLP, self).build(inputs_shape)
        
    
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        for num in range(1, self.num_blocks + 1):
            dense = self.get_layer("dense_{}".format(num))
            batch_norm = self.get_layer("batch_norm_{}".format(num))
            
            x = dense(x)
            x = batch_norm(x)
            x = self.dropout(x, training=training)
        output = self.classifier(x)
        return output
        