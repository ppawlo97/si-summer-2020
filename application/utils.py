"""Utility functions for image processing."""
from typing import Dict
from typing import List
from typing import Tuple

import cv2
import numpy as np

EMOTION_CMAP = {'neutral': (255, 255, 255),
                'happiness': (0, 150, 188),
                'surprise': (0, 140, 255),
                'sadness': (98, 98, 98),
                'anger': (21, 8, 200),
                'disgust': (65, 129, 0),
                'fear': (106, 0, 106),
                'contempt': (135, 184, 222),
                'unknown': (0, 0, 0)}


def extract_face(frame: np.ndarray,
                 bounding_box: Tuple) -> np.ndarray:
    """Crops the frame given bounding box."""
    x, y, w, h = bounding_box
    cropped_frame = frame[y:y+h, x:x+w]
    return cropped_frame


def draw_boxes(frame: np.ndarray,
               bounding_boxes: List[Tuple],
               predictions: List[Dict[str, float]]) -> np.ndarray:
    """Draws bounding boxes (with labels) on the frame."""
    if not predictions:
        predictions = [{"unknown": -1} for _ in range(len(bounding_boxes))]
    
    for bounding_box, prediction in zip(bounding_boxes, predictions):
        x, y, w, h = bounding_box
        label = max(prediction, key=prediction.get)
        color = EMOTION_CMAP[label]
        
        cv2.rectangle(frame,
                      (x, y),
                      (x + w, y + h),
                      color=color,
                      thickness=1)
        
        cv2.putText(img=frame,
                    text="{0}: {1:.2%}".format(label, prediction[label]),
                    org=(x + 5, y + h - 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=color,
                    thickness=1)
    return frame
