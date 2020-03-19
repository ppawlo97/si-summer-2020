from typing import List

import cv2
import numpy as np
from mtcnn import MTCNN

from application.models.DetectorInterface import DetectorInterface


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
        