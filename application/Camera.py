"""Contains frame streaming logic and applies selected models."""
import logging
from typing import Dict

import cv2

logger = logging.getLogger("Camera")


class Camera:
    def __init__(self):
        
        logger.info("Initializing camera...")
        self.camera = cv2.VideoCapture(0)

        
    @property
    def is_available(self):
        """Checks whether user's camera is available."""
        return (self.camera is not None) and (self.camera.isOpened())


    def generate(self,
                 models: Dict[str, str]) -> str:
        """Yields frames from stream."""
        stream = self.stream(**models)

        logger.info("Starting the stream...")
        for frame in stream:
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    def stream(self, detector: str, classifier: str) -> bytearray:
        """
        Continuously streams frames from an opened camera.
        Includes face detection and classification steps.
        """
        while self.camera.isOpened():
            status, img = self.camera.read()

            if not status:
                break
            
            yield cv2.imencode('.jpg', img)[1].tobytes()
