import os
from typing import List

import cv2
import numpy as np

from application.models.DetectorInterface import DetectorInterface


class CasClasDetector(DetectorInterface):
    def __init__(self,
                 pretrained_path: str):
        
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError("File with the pretrained\n\
                CascadeClassifier does not exists!")
        self.casclas = cv2.CascadeClassifier(pretrained_path)
        self.name = "casclas"

    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Overrides DetectorInterface.preprocess_frame()"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    def detect_faces(self, frame: np.ndarray) -> List[List[int]]:
        """Overrides DetectorInterface.detect_faces()"""
        res = self.casclas.detectMultiScale(frame,
                                            scaleFactor=1.1,
                                            minNeighbors=6)
        return res
        