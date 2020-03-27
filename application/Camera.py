"""Contains frame streaming logic and applies selected models."""
import logging
from typing import Dict

import cv2
import imutils
from imutils.video import WebcamVideoStream

from application.utils import draw_boxes
from application.utils import extract_face

logger = logging.getLogger("Camera")


class Camera:
    def __init__(self, 
                 detector: "Detector",
                 classifier: "Classifier"):
        
        self.detector = detector
        self.classifier = classifier
        logger.info("Initializing camera...")
        self.camera = WebcamVideoStream(src=0).start()


    def __del__(self): 
        logger.info("Releasing the camera...")
        self.camera.stop()
        self.camera.stream.release()
        
        
    @property
    def is_available(self) -> bool:
        """Checks whether user's camera is available."""
        return (self.camera is not None) and (self.camera.stream.isOpened())


    def generate(self) -> bytes:
        """Yields frames from stream."""
        stream = self.stream()

        logger.info("Starting the stream...")
        for frame in stream:
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    def stream(self) -> bytes:
        """
        Continuously streams frames from an opened camera.
        Includes face detection and classification steps.
        """
        while True:
            img = self.camera.read()
            img = imutils.resize(img, width=500, height=400)
            
            # Face Detection
            frame = self.detector.preprocess_frame(img)
            boxes = self.detector.detect_faces(frame)
            boxes = [box for box in boxes
                             if all(cord >= 0 for cord in box)] 
            
            faces = [extract_face(img, box) for box in boxes]
            
            # Emotion Recognition
            emotions = False
            if faces:
                images = self.classifier.preprocess_images(faces)
                emotions = self.classifier.predict_emotions(images)
            
            img = draw_boxes(img,
                             boxes,
                             emotions)

            yield cv2.imencode('.jpg', img)[1].tobytes()
