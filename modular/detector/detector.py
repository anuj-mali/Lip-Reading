
"""Defines a LandmarksDetector class that uses RetinaFace to detect faces
and return the landmarks of the face with biggest bounding box from the provided video file."""

import torch
from torchvision.io import read_video

from typing import List
import numpy as np

from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor

class LandmarksDetector:
    def __init__(self, device: torch.device = 'cpu'):
        self.face_detector = RetinaFacePredictor(device=device, threshold=0.8)
        self.landmark_detector = FANPredictor(device=device)

    def __call__(self, filename: str) -> List[np.ndarray]:
        frames = read_video(filename=str(filename), pts_unit='sec')[0].numpy()
        landmarks = []

        for frame in frames:
            face_boxes = self.face_detector(image=frame, rgb=False)
            face_points, _ = self.landmark_detector(image=frame, face_boxes=face_boxes)

            if len(face_boxes) == 0:
                landmarks.append(None)
            else:
                # Find the largest face box
                get_size = lambda box: box[2]-box[0] + box[3]-box[1]
                max_idx, max_face_box = max(enumerate(face_boxes), key=lambda x: get_size(x[1]))

                landmarks.append(face_points[max_idx])

        return landmarks
