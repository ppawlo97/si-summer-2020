"""
Convolutional Neural Network.

Notes 
-----
    1. CNN weights used in the project can be found in application/static/weights/cnn.
    2. Exact architecture of the CNN used in the project with the implementation
        can found at the bottom of the file. Use default parameters to obtain the same
        model as used in the project.
    3. If you want change the default LAB_MAP then application's __init__.py is
        the proper place to do it. 

"""
import os
from typing import Dict
from typing import List
from typing import Tuple

import cv2
import imutils
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D 

from application.models.classifiers.ClassifierInterface import ClassifierInterface
from application.models.classifiers.ClassifierInterface import LAB_MAP


class CNNClassifier(ClassifierInterface):
    def __init__(self,
                 cnn_path: str,
                 lab_map: Dict[str, int] = None):
        
        if not os.path.exists(cnn_path):
            raise FileNotFoundError("File with CNN\n\
                weights does not exist!")

        if lab_map is None:
            self.lab2idx = LAB_MAP
        else:
            self.lab2idx = lab_map

        self.idx2lab = {i: lab for lab, i in self.lab2idx.items()}
        self.cnn = tf.keras.models.load_model(cnn_path)
        self.name = "cnn"


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
        res = tf.dtypes.cast(res/255, dtype=tf.float32)
        return res


    def predict_emotions(self, images: tf.Tensor) -> List[Dict[str, float]]:
        """Overrides ClassifierInterface.predict_emotions()"""
        preds = self.cnn(images, training=False)
        res = [{self.idx2lab[ix]: p.numpy() for ix, p in enumerate(pred)}
                                                for pred in preds]
        return res


class CNN(tf.keras.Model):
    def __init__(self,
                 num_labels: int,
                 num_starting_filters: int = 128,
                 num_conv_stacks: int = 3,
                 kernel_size: Tuple[int, int] = (3, 3),
                 max_pool_size: Tuple[int, int] = (2, 2),
                 dropout_rate: float = 0.5,
                 num_dense_units: int = 512,
                 name: str= "cnn",
                 **kwargs):
        super(CNN, self).__init__(name=name, **kwargs)
        
        if num_conv_stacks < 1:
            raise Exception("Cannot have less than 1 stack!")
        self.num_conv_stacks = num_conv_stacks
        
        for num in range(1, num_conv_stacks + 1):
            setattr(self, "conv_{}".format(num), Conv2D(num_starting_filters*np.power(2, num-1),
                                                        kernel_size=kernel_size,
                                                        padding='valid', 
                                                        activation = 'relu', 
                                                        name="conv_{}".format(num)))
            setattr(self, "max_pooling_{}".format(num), MaxPooling2D(pool_size=max_pool_size,
                                                                     name="max_pooling_{}".format(num)))
            
        self.flatten = Flatten(name="flatten")
        self.dense = Dense(num_dense_units,
                           activation='relu', 
                           name="dense")
        self.dropout = Dropout(dropout_rate, 
                               name="dropout")
        self.classifier = Dense(num_labels, 
                                activation='softmax', 
                                name="classifier")
        
    
    def build(self, inputs_shape):
        super(CNN, self).build(inputs_shape)
        
    
    def call(self, inputs, training=False):
        x = inputs
        for num in range(1, self.num_conv_stacks + 1):
            conv = self.get_layer("conv_{}".format(num))
            max_pooling = self.get_layer("max_pooling_{}".format(num))    
            
            x = conv(x)
            x = max_pooling(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        output = self.classifier(x)
        return output
