"""
Utilizes pretrained MTCNN model.

References
----------
    1. Weights and implementation:
        https://github.com/ipazc/mtcnn
    2. Original paper:
        https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf

"""
from typing import List

import cv2
import numpy as np
from mtcnn import MTCNN

from application.models.detectors.DetectorInterface import DetectorInterface


class MTCNNDetector(DetectorInterface):
    def __init__(self):
        self.mtcnn = MTCNN()
        self.name = "mtcnn"
        
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Overrides DetectorInterface.preprocess_frame()"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    def detect_faces(self, frame: np.ndarray) -> List[List[int]]:
        """Overrides DetectorInterface.detect_faces()"""
        res = self.mtcnn.detect_faces(frame)
        boxes = [face["box"] for face in res]
        return boxes
        