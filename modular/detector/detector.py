"""Defines a LandmarksDetector class that uses RetinaFace to detect faces
and return the landmarks of the face with biggest bounding box from the provided video file."""

import torch
from torchvision.io import read_video

from typing import List
import numpy as np

from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor

import pandas as pd

import os

class LandmarksDetector:
    def __init__(self, device: torch.device = 'cpu'):
        self.face_detector = RetinaFacePredictor(device=device, threshold=0.8)
        self.landmark_detector = FANPredictor(device=device)

    def __call__(self, filename: str) -> List[np.ndarray]:
        # landmark_path = f"landmarks/computed/{str(filename).split('/')[-2]}/train/{str(filename).split('/')[-1].split('.')[-2]}.pkl"

        # if os.path.exists(landmark_path):
        #     return pd.read_pickle(landmark_path)
        frames = read_video(filename=str(filename), pts_unit='sec')[0].numpy()
        # frames = np.rot90(frames, k=1, axes=(1,2))
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
