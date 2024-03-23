# Dependencies
import torch
import numpy as np
from collections import deque
import json

from detector.detector import LandmarksDetector
from model.Resnet_GRU import ResNet_GRU
from utils.transform import VideoProcess

from typing import Dict

class Infer:
    def __init__(self,
                 model: torch.nn.Module,
                 landmarks_detector,
                 video_processor,
                 labels: Dict,
                 device: torch.device='cpu'):
        self.device = device
        self.landmarks_detector = landmarks_detector
        self.video_processor = video_processor
        self.model = model
        self.labels = labels

    async def __call__(self, path):
        model = self.model
        labels = self.labels
        device = self.device
        
        landmarks = self.landmarks_detector(filename=path)
        mouth_roi = self._pad(path, landmarks)

        result = {}
        model.eval()
        with torch.inference_mode():
                X = mouth_roi.unsqueeze(0).unsqueeze(0)
                y_pred = model(X.to(device))
                label = labels[torch.softmax(y_pred, dim=1).argmax().item()]
                confidence = torch.softmax(y_pred, dim=1).max().item()
                result['word'] = label
                result['confidence'] = max(confidence, result[label]) if result.get(label) else confidence
        return result

    def _pad(self, path, landmarks):
        mouth_roi = self.video_processor(path, landmarks)
        n = len(mouth_roi) if mouth_roi is not None else 0
        
        if n == 0:
            mouth_roi = np.zeros((29,96,96))
        elif n<29:
            padding = np.zeros((29-n, 96,96))
            mouth_roi = np.concatenate((mouth_roi, padding), axis=0)
        mouth_roi = torch.tensor(mouth_roi).to(torch.float32)
        return mouth_roi



def initialize(model_path: str='../checkpoint/ResNet_GRU_10.pth', 
               mean_landmarks_path: str='../mean_landmarks/20words_mean_face.npy',
               labels_path: str='labels.json') -> Infer:
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    
    landmarks_detector = LandmarksDetector(device=device)
    video_processor = VideoProcess(mean_landmarks_path=mean_landmarks_path)
    model = ResNet_GRU(in_channels=1, output_size=15, hidden_size=1664).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    with open(labels_path, 'r') as json_file:
        labels = json.load(json_file)

    return Infer(model=model,
                landmarks_detector=landmarks_detector,
                video_processor=video_processor,
                labels=labels,
                device=device)