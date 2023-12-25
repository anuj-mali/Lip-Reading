
import os
import dlib
from imutils import face_utils
import imutils
import numpy as np

landmarks_dir = "./landmarks/"

# Path of landmarks model
landmarks_path = os.path.join(landmarks_dir + "/shape_predictor_68_face_landmarks_GTX.dat")

# Create a face detector
face_detector = dlib.get_frontal_face_detector()

# Create a landmark detector
landmark_detector = dlib.shape_predictor(landmarks_path)

def get_landmarks( frame) -> np.ndarray:
    """Takes a frame and generates landmarks for the first face

        Args:
            frame: video frame or image required to generate landmarks
        
        Returns:
            A numpy array containing the co-ordinates of the landmarks of the first face in the given frame
    """
    faces = face_detector(frame)
    landmarks = None
    if faces:
        landmarks = landmark_detector(frame, faces[0])
        landmarks = face_utils.shape_to_np(landmarks)
    return landmarks

def generate_video_landmarks(frames) -> np.ndarray:
    """Generate landmarks the given video
    
        Args:
            filename (str): filename specifying the video
    
        Returns:
            A numpy.ndarray containing all the landmarks for the faces in the video"""
    landmarks = []
    for frame in frames:
        landmarks.append(get_landmarks(frame))
        landmarks = landmarks_interpolation(landmarks)
    return np.array(landmarks)

def landmarks_interpolation(landmarks):
    """Adds the missing landmarks to the landmarks array

    Args:
        landmarks: An array containing all the detected landmarks

    Returns:
        landmarks array filled in with missing landmarks
    """
    # Obtain indices of all the valid landmarks (i.e landmarks not None)
    valid_landmarks_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]

    # For middle parts of the landmarks array
    for idx in range(1, len(valid_landmarks_idx)):
        # If the valid landmarks indices are adjacent then skip to next iteration
        if valid_landmarks_idx[idx]-valid_landmarks_idx[idx-1] == 1:
            continue
        landmarks = linear_interpolation(start_idx=valid_landmarks_idx[idx-1],
                                        end_idx=valid_landmarks_idx[idx],
                                        landmarks=landmarks)

    # For beginning and ending parts of the landmarks array
    valid_landmarks_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    landmarks[:valid_landmarks_idx[0]] = [landmarks[valid_landmarks_idx[0]]] * valid_landmarks_idx[0]
    landmarks[valid_landmarks_idx[-1]:] = [landmarks[valid_landmarks_idx[-1]]] * (len(landmarks) - valid_landmarks_idx[-1])

    return landmarks

def linear_interpolation(start_idx: int, end_idx: int, landmarks):
    """Defines a linear interpolation function to interpolate missing landmarks between indices

    Args:
        start_idx (int): An integer defining the starting index
        end_idx (int): An integer defining the stopping index
        landmarks: An array of size 68 containing the (x,y) values of the facial landmarks

    Returns:
        landmarks array after the missing points have been interpolated.
    """
    start_landmarks = landmarks[start_idx]
    end_landmarks = landmarks[end_idx]
    delta_idx = end_idx - start_idx
    delta_landmarks = end_landmarks - start_landmarks

    # Apply linear interpolation formula
    for idx in range(1, delta_idx):
        landmarks[idx + start_idx] = start_landmarks + delta_landmarks/delta_idx * idx
    return landmarks

def find_classes(dataframe: pd.DataFrame) -> Tuple[List, Dict[str, int]]:
    class_names = dataframe['label'].unique()
    class_to_idx = {label: idx for idx, label in enumerate(class_names)}
    return class_names, class_to_idx
