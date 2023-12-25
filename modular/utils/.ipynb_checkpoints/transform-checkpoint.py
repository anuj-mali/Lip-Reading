import cv2
import numpy as np
from torchvision.io import read_video
from .interpolation import landmarks_interpolation

class VideoProcess:
    def __init__(self, mean_landmarks_path: str):
        # For warping the frames
        self.mean_landmarks = np.load(mean_landmarks_path)
        self.stable_points=(28, 33, 36, 39, 42, 45, 48, 54)
        self.reference_landmarks = np.vstack([self.mean_landmarks[x] for x in self.stable_points])

        # Cropping
        self.crop_height = 96
        self.crop_width = 96
        self.temporal_window = 12

    def __call__(self, video_path, landmarks):
        video,_,_ = read_video(str(video_path), pts_unit='sec')
        landmarks = landmarks_interpolation(landmarks)

        if not landmarks or len(landmarks) < self.temporal_window:
            return 

        mouth_rois = self.crop_video(video.numpy(), landmarks)
        return mouth_rois

    
    # Affine Transformation (Rotation, Translation and Scaling)
    def _get_transform(self, src_points: np.ndarray) -> np.ndarray:
        transform, _ = cv2.estimateAffinePartial2D(src_points, self.reference_landmarks, method=cv2.LMEDS)
        return transform

    def warp_frame(self, frame, landmarks):
        # Convert frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Extract only stable points from current landmarks
        stable_points_landmarks = np.vstack([landmarks[x] for x in self.stable_points])

        transform_matrix = self._get_transform(stable_points_landmarks)
        transformed_frame = cv2.warpAffine(frame, transform_matrix, dsize=(256,256))
        transformed_landmarks = np.matmul(landmarks, transform_matrix[:, :2].transpose()) + transform_matrix[:, 2].transpose()
        
        return transformed_frame, transformed_landmarks


    # Crop Frames (Obtain Mouth ROI)
    def _cut_mouth(self, frame, landmarks, height, width):
        center_x, center_y = np.mean(landmarks, axis=0)

        # Calculate bounding box coordinates
        y_min = int(round(np.clip(center_y - height, 0, 256)))
        y_max = int(round(np.clip(center_y + height, 0, 256)))
        x_min = int(round(np.clip(center_x - width, 0, 256)))
        x_max = int(round(np.clip(center_x + width, 0, 256)))
        mouth_roi = frame[y_min:y_max, x_min:x_max]
        return mouth_roi

    def crop_video(self, video, landmarks_list):
        mouth_rois = []
        for idx, frame in enumerate(video):
            window_margin = min(self.temporal_window//2, idx, len(landmarks_list)-idx-1)

            average_landmarks_temporal = np.mean([landmarks_list[x] for x in range(idx-window_margin, idx+window_margin+1)], axis=0)
            average_landmarks_temporal += landmarks_list[idx].mean(axis=0) - average_landmarks_temporal.mean(axis=0)

            transformed_frame, transformed_landmarks = self.warp_frame(frame, average_landmarks_temporal)

            mouth = self._cut_mouth(transformed_frame, transformed_landmarks[48:68], height=self.crop_height//2, width=self.crop_width//2)

            mouth_rois.append(mouth)
        # print(mouth_rois)
        return np.array(mouth_rois)
