from abc import ABCMeta
from abc import abstractmethod
from typing import List

import numpy as np


class DetectorInterface(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'preprocess_frame') and 
                callable(subclass.preprocess_frame) and 
                hasattr(subclass, 'detect_faces') and 
                callable(subclass.detect_faces) or 
                NotImplemented)


    @abstractmethod
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocesses incoming frame before prediction."""
        raise NotImplementedError


    @abstractmethod
    def detect_faces(self, frame: np.ndarray) -> List[List[int]]:
        """Detects faces in the image."""
        raise NotImplementedError
